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

### Two-branch teacher-student (v3)

Two masking levels: (1) word/sentence masking within paragraphs (X0→Xm), (2) paragraph dropout at pooling stage. Both branches share one `distilroberta-base` encoder. Teacher pass uses `stop_grad=True`. Two separate `AttentionPooling` modules — student trains via gradient, teacher updates via EMA (τ=0.99).

```
X0 (clean) → encoder (stop-grad) → N teacher reps → attn_pool_teacher (EMA) → Z_prime (768)
Xm (masked) → encoder (trainable) → N student reps
                                          ↓ paragraph dropout (30%) → M reps
                                    attn_pool_student → z_partial (768)
                                          ↓ IB head: mu/sigma (256), reparameterize
                                    projector (256→512→768) → Z_proj (768)
```

`_encode_paragraphs` batches all sub-chunks in one encoder forward pass. Paragraphs >510 tokens are split into sub-chunks, encoded, and mean-pooled into one vector. Documents with >50 paragraphs are excluded at data loading (not truncated).

### Four-component loss with Homoscedastic Uncertainty Weighting

```
L = (L_compress · exp(−s₀) + s₀)   # KL( N(μ,σ²) ∥ N(0,1) )
  + (L_local   · exp(−s₁) + s₁)   # cosine dist: all N student para reps vs teacher para reps
  + (L_inter   · exp(−s₂) + s₂)   # cosine dist: student M-para pool vs teacher full-doc (pre-IB)
  + (L_global  · exp(−s₃) + s₃)   # cosine dist: Z_proj vs Z_prime (post-IB)

log_s = nn.Parameter(torch.zeros(4))   # clamped to [-10, 10]
```

EMA update: `model.update_teacher_ema()` called after every `optimizer.step()`.

### Data flow through training

`mask_text(paragraph)` → word/sentence masked string (Xm per paragraph).
`get_paragraph_mask(n)` → kept indices list (30% dropped).
`load_ecthr()` filters documents: 5 ≤ paragraphs ≤ 50.
`sample_few_shot(split, n_per_class, seed)` is multi-label aware — deduplicates.

### Checkpoint format

`scripts/02_pretrain_glocal.py` saves two files per epoch:
- `checkpoints/{CONDITION}_epoch{N}/` — full accelerator state (resumable)
- `checkpoints/{CONDITION}_epoch{N}.pt` — plain `model.state_dict()` (used by fine-tuning)

MLM baseline saves HuggingFace format: `checkpoints/mlm_baseline/`.

## src/ API

**`src/data.py`**
- `load_ecthr(min_paragraphs=5, max_paragraphs=50)` → `DatasetDict`
- `mask_text(paragraph, sent_dropout=0.20, span_rate=0.15)` → `str`
- `get_paragraph_mask(n_paragraphs, dropout_rate=0.30)` → `list[int]`
- `mask_paragraphs(paragraphs)` → `(list[str], list[int])` (legacy, still works)
- `sample_few_shot(split, n_per_class, seed, num_classes=10)` → `list[dict]`

**`src/model.py`**
- `GlocalIBModel(hidden_dim=256, proj_dim=512, max_paragraphs=50, ema_tau=0.99, device="cuda")`
  - forward: `(full_batch, masked_batch, kept_indices_batch)` → 8-tuple `(Z_prime, Z_proj, z_partial, s_chunks, t_chunks, mu, sigma, log_s)`
  - `update_teacher_ema()` — call after every optimizer step
- `DocumentClassifier(encoder, tokenizer, attn_pool, num_labels=10, max_paragraphs=50, device="cuda")` — forward: `(paragraphs_batch)` → `(B, 10)` sigmoid
- `AttentionPooling(dim=768, max_chunks=50)`

**`src/loss.py`**
- `glocal_ib_loss(Z_prime, Z_proj, z_partial, s_chunks, t_chunks, mu, sigma, log_s)` → `(total, l_compress, l_local, l_inter, l_global)`
- `alignment_loss(z1, z2)`, `compression_loss(mu, sigma)`

## Experimental Conditions

| Condition | Pre-training |
|-----------|-------------|
| `glocal_ib` | Full 4-component loss + UW + EMA attention |
| `mlm` | Standard MLM on ECtHR (vanilla 15% token masking) |
| `mlm` | Standard MLM on ECtHR paragraphs | N/A |

Fine-tuning: N ∈ {10, 50, 100} × 5 seeds. Metric: **macro-F1** (mandatory — class imbalance is severe: label 3 has 4704 training examples, label 5 has 41).

## Key Invariants

- `distilroberta-base` loads via `RobertaModel.from_pretrained("distilroberta-base")` — no `DistilRobertaModel`.
- Teacher encoder pass is always `stop_grad=True` (`torch.no_grad()` inside `_encode_paragraphs`).
- Teacher/student encoding passes are sequential — never simultaneous — to minimize peak VRAM.
- `log_s` clamped to `[-10, 10]` in every forward pass (prevents numerical explosion).
- `update_teacher_ema()` must be called after every `optimizer.step()` — not inside `forward`.
- If all four `log_s` stay near 0 through epoch 2, UW is degenerate — flag it, don't ignore.
- `scripts/` contains standalone Python equivalents of all notebooks (SSH/cluster friendly). Notebooks in `notebooks/` are kept for interactive use.

## Compute

| Job | Recommended | Batch config |
|-----|------------|-------------|
| GlocalIB pre-training | 4× A100 (Spark) | `BATCH_SIZE=1`, `GRAD_ACCUM=4` → effective 16 |
| MLM pre-training | 4× A100 (Spark) | `BATCH_SIZE=4` |
| Fine-tuning / exploration | 1× 32GB GPU | `BATCH_SIZE=4` |
| Sanity checks | CPU | no GPU required |

Fallback cluster: 10× NVIDIA TITAN Xp is usable but slower and memory-constrained
(12GB VRAM, no bf16). Use `BATCH_SIZE=1`, `GRAD_ACCUM=4`, and change Accelerate
mixed precision from `bf16` to `fp16` or disable mixed precision. If GlocalIB still
OOMs, sub-batch `_encode_paragraphs()` instead of reducing the 50-paragraph cap.
W&B project: `glocal-nlp`.
