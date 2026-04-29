# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MSc research project at Free University of Bozen-Bolzano (April 2026). Adapts the GlocalIB (Global-Local Information Bottleneck) objective from time series imputation to few-shot legal document classification using the ECtHR dataset.

The core bet: forcing a model to reconstruct a full document's meaning from partially masked paragraphs — with an explicit compression bottleneck — produces representations that generalize better with only 10–100 labeled examples.

See `doc/TODO.md` for current implementation status and `doc/PLAN.md` for the full step-by-step implementation plan.

## Environment Setup

```bash
conda create -n glocal_nlp python=3.10
conda activate glocal_nlp
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

Verify GPU and imports:
```bash
python -c "import torch; print(torch.cuda.is_available())"
python -c "from src.data import load_ecthr; from src.model import GlocalIBModel; from src.loss import glocal_ib_loss; print('OK')"
```

## Compute

| Job | Node | SLURM flags |
|-----|------|-------------|
| GlocalIB pre-training | Spark 128GB | `--partition=spark --gres=gpu:4` |
| MLM pre-training | Spark 128GB | `--partition=spark --gres=gpu:4` |
| Fine-tuning / exploration | Standard 32GB | `--gres=gpu:1` |

Batch size guidance: `BATCH_SIZE=1` + `GRAD_ACCUM=4` on 32GB, `BATCH_SIZE=4` + no grad accum on Spark. Confirm the Spark partition name with the lab admin before submitting.

## src/ Module API

All notebooks import from `src/` via `sys.path.append("..")`.

**`src/data.py`**
- `load_ecthr(min_paragraphs=5)` — loads `coastalcph/lex_glue / ecthr_a` from HuggingFace, filters docs with fewer than 5 paragraphs. Returns a HuggingFace `DatasetDict`.
- `mask_paragraphs(paragraphs, mask_ratio_min=0.2, mask_ratio_max=0.4)` — randomly drops 20–40% of paragraphs, always keeps at least 1. Returns `(kept_paragraphs: list[str], kept_indices: list[int])`.
- `sample_few_shot(dataset_split, n_per_class, seed, num_classes=10)` — multi-label aware: samples until each class has ≥ n_per_class examples, deduplicates documents that satisfy multiple classes.

**`src/loss.py`**
- `alignment_loss(z1, z2)` — `1 - mean cosine_similarity(z1, z2)`, inputs (N, D).
- `compression_loss(mu, sigma)` — closed-form KL(N(mu, sigma²) ∥ N(0,1)), inputs (B, H).
- `glocal_ib_loss(Z_prime, Z_proj, Z_inter_s, Z_inter_t, chunks_s, chunks_t, mu, sigma, log_s, disable_ib=False)` — returns `(total, l_compress, l_local, l_inter, l_global)`. Pass `disable_ib=True` for the `glocal_beta0` ablation.

**`src/model.py`**
- `AttentionPooling(dim=768, max_chunks=200)` — learnable query-based pooling with positional embeddings. Zero-initialized (starts as mean pooling).
- `GlocalIBModel(hidden_dim=256, proj_dim=512, max_chunks=50, device="cuda")` — full teacher-student architecture using `distilroberta-base` (loaded as `RobertaModel`). Forward: `forward(full_batch, masked_batch, kept_indices_batch)` where all args are `list[list[str]]` / `list[list[int]]`. Returns `(Z_prime, Z_proj, Z_inter_s, Z_inter_t, chunks_s, chunks_t, mu, sigma, log_s)`.
- `DocumentClassifier(encoder, tokenizer, num_labels=10, max_chunks=50, device="cuda")` — fine-tuning classifier. Takes a pre-trained encoder. Forward: `forward(paragraphs_batch: list[list[str]])` → `(B, 10)` sigmoid probabilities.

## Architecture (v2)

Two-branch setup using `distilroberta-base` weights (6-layer RoBERTa, 82M params):

**Teacher branch** (stop-gradient): batched encode all paragraphs → `AttentionPooling` → deterministic Z' (768-dim).

**Student branch** (trainable): batched encode kept paragraphs → `AttentionPooling` → IB bottleneck → mu (256), sigma (256) → reparameterization → MLP projector → Z_proj (768).

```
Probabilistic head:   z_partial (768) → Linear → mu (256)
                      z_partial (768) → Linear → log_sigma → sigma = exp(log_sigma)
                      Z = mu + sigma * N(0,1)

MLP projector:        Z (256) → Linear(256,512) → ReLU → Linear(512,768) → Z_proj

Loss weights:         self.log_s = nn.Parameter(torch.zeros(4))
                      weights[i] = exp(-log_s[i])   (Homoscedastic Uncertainty Weighting)
```

**Batched encoding**: all paragraphs tokenized in a single forward pass with dynamic padding (`padding=True`). Combined with `distilroberta-base` and gradient checkpointing, this fits on a 32GB GPU at `BATCH_SIZE=1`.

## Loss (v2)

```
L_total = L_compress × exp(−s₀) + s₀
        + L_local   × exp(−s₁) + s₁
        + L_inter   × exp(−s₂) + s₂
        + L_global  × exp(−s₃) + s₃

L_compress  = KL( N(mu, sigma²) || N(0,1) )
L_local     = 1 - cosine_sim(student_chunks_kept, teacher_chunks_kept)
L_inter     = 1 - cosine_sim(attn_pool(s_chunks), attn_pool(t_chunks_kept))
L_global    = 1 - cosine_sim(Z_proj, Z_prime)
```

## Experimental Conditions

Three conditions — identical data, identical starting weights, only the pre-training objective differs:

| Condition | Description | `disable_ib` |
|---|---|---|
| `mlm` | DistilRoBERTa + standard MLM on ECtHR (HF Trainer) | N/A |
| `glocal_beta0` | Full GlocalIB architecture, global alignment only (ablation) | `True` |
| `glocal_ib` | Full 4-component loss + Homoscedastic UW | `False` |

Fine-tuning: N = {10, 50, 100} labeled examples × 5 seeds. Metric: macro-F1 on ECtHR test set.

## Key Invariants

- Teacher always sees the full document. Stop-gradient enforced via `torch.no_grad()` in `_encode_chunks(stop_grad=True)`.
- Student paragraph masking is independent of the 512-token chunk limit — they are separate mechanisms.
- `distilroberta-base` loads as `RobertaModel` (no `DistilRobertaModel` class in transformers).
- Evaluation metric must be macro-F1 (heavy class imbalance: Article 3 has 4704 cases, Article 5 has 41).
- If all four `log_s` stay near 0 by epoch 2, the uncertainty weighting is degenerate — report as diagnostic.

## Notebooks Workflow

Notebooks live in `notebooks/` and import from `src/` via `sys.path.append("..")`. Run on the cluster via `jupyter nbconvert --to notebook --execute`. SLURM job scripts are in `slurm/`.

| Notebook | Purpose |
|---|---|
| `01_data_exploration.ipynb` | Dataset stats, masking sanity check |
| `02_pretrain_glocal.ipynb` | GlocalIB pre-training (set `CONDITION` in Cell 1) |
| `03_pretrain_mlm.ipynb` | MLM baseline via HuggingFace Trainer |
| `04_finetune.ipynb` | All 3 conditions × N × seeds → `results/finetuning_results.json` |
| `05_evaluate.ipynb` | Macro-F1 table, performance curve, log_s trajectory from W&B |

## Logging

W&B project: `glocal-nlp`. Log per training step: `total_loss`, `l_compress`, `l_local`, `l_inter`, `l_global`, `log_s_0_compress`, `log_s_1_local`, `log_s_2_inter`, `log_s_3_global`, `epoch`. The `log_s` trajectory is the key diagnostic — fetch it in `05_evaluate.ipynb` via `wandb.Api()`.
