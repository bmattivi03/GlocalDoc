# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MSc research project at Free University of Bozen-Bolzano (April 2026). Adapts the GlocalIB (Global-Local Information Bottleneck) objective from time series imputation to few-shot legal document classification using the ECtHR dataset.

The core bet: forcing a model to reconstruct a full document's meaning from partially masked paragraphs — with an explicit compression bottleneck — produces representations that generalize better with only 10–100 labeled examples.

See `TODO.md` for current implementation status and `PLAN.md` for the full step-by-step implementation plan.

## Environment Setup

```bash
conda create -n glocal_nlp python=3.10
conda activate glocal_nlp
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
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

Batch size guidance: `BATCH_SIZE=4` on 32GB, `BATCH_SIZE=16` on Spark. Confirm the Spark partition name with the lab admin before submitting.

## src/ Module API

All notebooks import from `src/` via `sys.path.append("..")`.

**`src/data.py`**
- `load_ecthr(min_paragraphs=5)` — loads `coastalcph/lex_glue / ecthr_a` from HuggingFace, filters docs with fewer than 5 paragraphs. Returns a HuggingFace `DatasetDict`.
- `mask_paragraphs(paragraphs, mask_ratio_min=0.2, mask_ratio_max=0.4)` — randomly drops 20–40% of paragraphs, always keeps at least 1. Used to produce the student branch input.
- `sample_few_shot(dataset_split, n_per_class, seed, num_classes=10)` — multi-label aware: samples until each class has ≥ n_per_class examples, deduplicates documents that satisfy multiple classes.

**`src/loss.py`**
- `alignment_loss(z_proj, z_prime)` — `1 - cosine_similarity(Z_proj, Z')`, both (B, 768).
- `compression_loss(mu, sigma)` — closed-form KL(N(mu, sigma²) ∥ N(0,1)).
- `glocal_ib_loss(z_proj, z_prime, mu, sigma, beta, disable_ib=False)` — returns `(total, l_align, l_compress)`. Pass `disable_ib=True` for the β=0 ablation condition.

**`src/model.py`**
- `GlocalIBModel(hidden_dim=256, proj_dim=512, device="cuda")` — full teacher-student architecture. Forward signature: `forward(full_paragraphs_batch, masked_paragraphs_batch)` where both args are `list[list[str]]`. Returns `(Z_prime, mu, sigma, Z_proj, beta)`.
- `DocumentClassifier(encoder, tokenizer, num_labels=10, device="cuda")` — fine-tuning classifier. Takes a pre-trained encoder (extracted from `GlocalIBModel` or `RobertaForMaskedLM`). Forward: `forward(paragraphs_batch: list[list[str]])` → `(B, 10)` sigmoid probabilities.

## Architecture

Two-branch setup sharing RoBERTa-base weights:

**Teacher branch** (stop-gradient): encodes the full document → deterministic Z' (768-dim).

**Student branch** (trainable): encodes the masked document → probabilistic head → mu (256), sigma (256) → reparameterization → MLP projector → Z_proj (768).

```
Probabilistic head:   CLS (768) → Linear → mu (256)
                      CLS (768) → Linear → log_sigma → sigma = exp(log_sigma)
                      Z = mu + sigma * N(0,1)

MLP projector:        Z (256) → Linear(256,512) → ReLU → Linear(512,768) → Z_proj

Learnable beta:       log_beta = nn.Parameter(torch.tensor(0.0))
                      beta = clamp(exp(log_beta), min=0.01)
```

**Chunk-and-pool**: each paragraph is encoded independently via RoBERTa CLS token (768-dim), then mean-pooled into one document vector. Handles RoBERTa's 512-token limit without requiring a long-context model.

## Loss

```
L = L_alignment + beta * L_compression
L_alignment   = 1 - cosine_similarity(Z_proj, Z')
L_compression = KL( N(mu, sigma^2) || N(0,1) )
              = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
```

## Experimental Conditions

Three conditions — identical data, identical starting weights, only the pre-training objective differs:

| Condition | Description |
|---|---|
| `mlm` | RoBERTa + standard MLM on ECtHR (HF Trainer) |
| `glocal_beta0` | Full GlocalIB architecture, `disable_ib=True` (ablation) |
| `glocal_ib` | Full GlocalIB with learnable beta (our method) |

Fine-tuning: N = {10, 50, 100} labeled examples × 5 seeds. Metric: macro-F1 on ECtHR test set.

## Key Invariants

- Teacher always sees the full document. Stop-gradient is enforced via `torch.no_grad()` in `_encode_paragraphs(stop_grad=True)`.
- Student paragraph masking is independent of the 512-token chunk limit — they are separate mechanisms.
- Beta lower bound of 0.01 prevents compression collapse — do not remove the clamp.
- Evaluation metric must be macro-F1 (heavy class imbalance: Article 3 has 4704 cases, Article 5 has 41).
- If beta collapses to 0.01 throughout training, the IB term is not doing real work — report this as a diagnostic, not a silent failure.

## Notebooks Workflow

Notebooks live in `notebooks/` and import from `src/` via `sys.path.append("..")`. Run on the cluster via `jupyter nbconvert --to notebook --execute`. SLURM job scripts are in `slurm/`.

| Notebook | Purpose |
|---|---|
| `01_data_exploration.ipynb` | Dataset stats, masking sanity check |
| `02_pretrain_glocal.ipynb` | GlocalIB pre-training (set `CONDITION` in Cell 1) |
| `03_pretrain_mlm.ipynb` | MLM baseline via HuggingFace Trainer |
| `04_finetune.ipynb` | All 3 conditions × N × seeds → `results/finetuning_results.json` |
| `05_evaluate.ipynb` | Macro-F1 table, performance curve, beta trajectory from W&B |

## Logging

W&B project: `glocal-nlp`. Log per training step: `loss`, `l_align`, `l_compress`, `beta`, `epoch`. Beta trajectory is the key diagnostic — fetch it in `05_evaluate.ipynb` via `wandb.Api()`.
