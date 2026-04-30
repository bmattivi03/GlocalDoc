# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MSc research at Free University of Bozen-Bolzano (2026). Adapts the GlocalIB (Global-Local Information Bottleneck) objective to few-shot legal document classification on the ECtHR dataset (`coastalcph/lex_glue / ecthr_a`). The core hypothesis: compressing masked-document representations through an explicit IB bottleneck produces better few-shot features than standard MLM pre-training.

Full architecture spec and rationale: `doc/PLAN.md`. Task status: `doc/TODO.md`.

## Common Commands

```bash
# Verify all imports and a full forward+backward pass (CPU, no GPU needed)
python -c "
import torch; from src.model import GlocalIBModel; from src.loss import glocal_ib_loss; from src.data import mask_paragraphs
model = GlocalIBModel(device='cpu')
full = [['Para A.', 'Para B.', 'Para C.', 'Para D.']]
md = [mask_paragraphs(p) for p in full]
out = model(full, [m[0] for m in md], [m[1] for m in md])
total, *_ = glocal_ib_loss(*out)
total.backward()
print('OK', total.item())
"

# Data exploration (prints stats, saves histogram to results/)
python scripts/01_data_exploration.py

# Pre-train — single GPU
python scripts/02_pretrain_glocal.py        # edit CONDITION at top first

# Pre-train — multi-GPU (e.g. 10 GPUs)
accelerate launch --num_processes=10 scripts/02_pretrain_glocal.py

# MLM baseline
accelerate launch --num_processes=10 scripts/03_pretrain_mlm.py

# Fine-tune all 3 conditions × N × seeds → results/finetuning_results.json
python scripts/04_finetune.py
```

## Architecture

### Two-branch teacher-student

Both branches share a single `distilroberta-base` encoder (loaded as `RobertaModel` — there is no `DistilRobertaModel` class in transformers). The teacher sees the **full** document; the student sees only the **kept** paragraphs after masking.

```
Teacher (stop-grad):  paragraphs → _encode_chunks(stop_grad=True) → AttentionPooling → Z′  (768)
Student (trainable):  kept paras → _encode_chunks(stop_grad=False) → AttentionPooling → z_partial (768)
                                                                                               ↓
                                                              mu_head / log_sigma_head → μ, σ  (256)
                                                              reparameterize → z_sample
                                                              projector (256→512→768) → Z_proj  (768)
```

`_encode_chunks` does a **single batched forward pass** over all paragraphs (dynamic padding, `padding=True`). Documents longer than `max_chunks=50` are truncated symmetrically (first half + last half).

`AttentionPooling` uses a learnable query vector + positional embeddings (zero-initialized → starts as mean pooling, learns deviations).

### Four-component loss with Homoscedastic Uncertainty Weighting

```
L = (L_compress · exp(−s₀) + s₀)   # KL( N(μ,σ²) ∥ N(0,1) )
  + (L_local   · exp(−s₁) + s₁)   # cosine dist: student chunks vs teacher chunks (kept positions)
  + (L_inter   · exp(−s₂) + s₂)   # cosine dist: student partial pool vs teacher partial pool
  + (L_global  · exp(−s₃) + s₃)   # cosine dist: Z_proj vs Z′

log_s = nn.Parameter(torch.zeros(4))   # all four weights learned jointly
```

`disable_ib=True` (for `glocal_beta0` condition) skips everything and returns only `L_global`.

### Data flow through training

`mask_paragraphs(paragraphs)` → returns `(kept_paragraphs: list[str], kept_indices: list[int])`.
`load_ecthr()` filters documents with fewer than 5 paragraphs.
`sample_few_shot(split, n_per_class, seed)` is multi-label aware — deduplicates docs satisfying multiple classes.

### Checkpoint format

`train_glocal.py` (and `scripts/02_pretrain_glocal.py`) saves **two** files per epoch:
- `checkpoints/{CONDITION}_epoch{N}/` — full accelerator state (resumable training)
- `checkpoints/{CONDITION}_epoch{N}.pt` — plain `model.state_dict()` (used by `scripts/04_finetune.py`)

MLM baseline saves in HuggingFace format via `trainer.save_model("checkpoints/mlm_baseline")`.

## src/ API

**`src/data.py`** — do not modify
- `load_ecthr(min_paragraphs=5)` → `DatasetDict`
- `mask_paragraphs(paragraphs)` → `(list[str], list[int])`
- `sample_few_shot(split, n_per_class, seed, num_classes=10)` → `list[dict]`

**`src/model.py`**
- `GlocalIBModel(hidden_dim=256, proj_dim=512, max_chunks=50, device="cuda")` — forward: `(full_batch, masked_batch, kept_indices_batch)` → 9-tuple `(Z_prime, Z_proj, Z_inter_s, Z_inter_t, chunks_s, chunks_t, mu, sigma, log_s)`
- `DocumentClassifier(encoder, tokenizer, num_labels=10, max_chunks=50, device="cuda")` — forward: `(paragraphs_batch)` → `(B, 10)` sigmoid
- `AttentionPooling(dim=768, max_chunks=200)`

**`src/loss.py`**
- `glocal_ib_loss(*model_output, disable_ib=False)` → `(total, l_compress, l_local, l_inter, l_global)`
- `alignment_loss(z1, z2)`, `compression_loss(mu, sigma)`

## Experimental Conditions

| Condition | Pre-training | `disable_ib` |
|-----------|-------------|--------------|
| `glocal_ib` | Full 4-component loss + UW | `False` |
| `glocal_beta0` | Global alignment only (ablation) | `True` |
| `mlm` | Standard MLM on ECtHR paragraphs | N/A |

Fine-tuning: N ∈ {10, 50, 100} × 5 seeds. Metric: **macro-F1** (mandatory — class imbalance is severe: label 3 has 4704 training examples, label 5 has 41).

## Key Invariants

- `distilroberta-base` loads via `RobertaModel.from_pretrained("distilroberta-base")` — no `DistilRobertaModel`.
- Teacher branch is always `stop_grad=True` (`torch.no_grad()` inside `_encode_chunks`).
- `log_s` uses Kendall & Gal Homoscedastic UW — no `clamp` on weights.
- If all four `log_s` stay near 0 through epoch 2, the uncertainty weighting is degenerate — flag it, don't ignore.
- `scripts/` contains standalone Python equivalents of all notebooks (SSH/cluster friendly). Notebooks in `notebooks/` are kept for interactive use.

## Compute

| Job | Recommended | Batch config |
|-----|------------|-------------|
| GlocalIB pre-training | 4× A100 (Spark) | `BATCH_SIZE=1`, `GRAD_ACCUM=4` → effective 16 |
| MLM pre-training | 4× A100 (Spark) | `BATCH_SIZE=4` |
| Fine-tuning / exploration | 1× 32GB GPU | `BATCH_SIZE=4` |
| Sanity checks | CPU | no GPU required |

On 10× 11GB GPUs: keep `BATCH_SIZE=1`, `GRAD_ACCUM=4` → effective batch 40. W&B project: `glocal-nlp`.
