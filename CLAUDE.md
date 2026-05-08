# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MSc research at Free University of Bozen-Bolzano (2026). Adapts the GlocalIB (Global-Local Information Bottleneck) objective to few-shot legal document classification on the ECtHR dataset (`coastalcph/lex_glue / ecthr_a`). The core hypothesis: compressing masked-document representations through an explicit IB bottleneck produces better few-shot features than standard MLM pre-training.

Full architecture spec and rationale: `doc/PLAN.md`. Task status: `doc/TODO.md`.

## Common Commands

```bash
# GlocalIB sanity check — full forward+backward pass (CPU, no GPU needed)
python -c "
import torch
from src.model import GlocalIBModel
from src.loss import glocal_ib_loss
from src.data import mask_text, get_paragraph_mask
model = GlocalIBModel(device='cpu')
full = [['Para A.', 'Para B.', 'Para C.', 'Para D.']]
masked_batch       = [[mask_text(p) for p in doc] for doc in full]
kept_indices_batch = [get_paragraph_mask(len(doc)) for doc in full]
out = model(full, masked_batch, kept_indices_batch)
total, *_ = glocal_ib_loss(*out)
total.backward()
print('GlocalIB OK', total.item())
"

# H-MLM sanity check — forward+backward + checkpoint round-trip (CPU)
python -c "
import torch, sys; sys.path.append('.')
from transformers import RobertaForMaskedLM, RobertaTokenizerFast, DataCollatorForLanguageModeling, RobertaModel
from src.model import AttentionPooling, DocumentClassifier
from src.loss import alignment_loss
from src.data import get_paragraph_mask
tokenizer = RobertaTokenizerFast.from_pretrained('distilroberta-base')
model     = RobertaForMaskedLM.from_pretrained('distilroberta-base')
attn_pool = AttentionPooling(dim=768, max_chunks=50)
mlm_coll  = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15, return_tensors='pt')
paragraphs = ['Para A.', 'Para B.', 'Para C.']
para_ids = [{'input_ids': tokenizer.encode(p, truncation=True, max_length=512)} for p in paragraphs]
l_mlm = model(**mlm_coll(para_ids)).loss
with torch.no_grad():
    t_cls = model.roberta(**tokenizer(paragraphs, padding=True, truncation=True, max_length=512, return_tensors='pt')).last_hidden_state[:, 0, :]
kept  = get_paragraph_mask(len(paragraphs))
s_cls = model.roberta(**tokenizer([paragraphs[i] for i in kept], padding=True, truncation=True, max_length=512, return_tensors='pt')).last_hidden_state[:, 0, :]
(0.5 * l_mlm + 0.5 * alignment_loss(attn_pool(s_cls).unsqueeze(0), t_cls.mean(0).detach().unsqueeze(0))).backward()
print('H-MLM OK')
"

# Data exploration (prints stats, saves histogram to results/)
python scripts/01_data_exploration.py

# Pre-train GlocalIB — single GPU
python scripts/02_pretrain_glocal.py        # edit CONDITION at top first

# Pre-train GlocalIB — multi-GPU (e.g. 4 GPUs)
accelerate launch --num_processes=4 scripts/02_pretrain_glocal.py

# Pre-train H-MLM baseline
accelerate launch --num_processes=4 scripts/03_pretrain_mlm.py

# Fine-tune all conditions × N × seeds → results/finetuning_results.json
python scripts/04_finetune.py
```

## Architecture

### GlocalIB: Two-branch teacher-student (v3)

Two masking levels: (1) word/sentence masking within paragraphs (`mask_text`), (2) paragraph dropout at pooling stage (`get_paragraph_mask`). Both branches share one `distilroberta-base` encoder. Teacher pass uses `stop_grad=True`. Two separate `AttentionPooling` modules — student trains via gradient, teacher updates via EMA (τ=0.99).

```
X0 (clean) → encoder (stop-grad) → N teacher reps → attn_pool_teacher (EMA) → Z_prime (768)
Xm (masked) → encoder (trainable) → N student reps
                                          ↓ paragraph dropout (30%) → M reps
                                    attn_pool_student → z_partial (768)
                                          ↓ IB head: mu/sigma (256), reparameterize
                                    projector (256→512→768) → Z_proj (768)
```

`_encode_paragraphs` batches all sub-chunks in one encoder forward pass. Paragraphs >510 tokens are split into sub-chunks, encoded, and mean-pooled into one vector. Documents with >50 paragraphs are excluded at data loading (not truncated).

**Critical:** `masked_batch` must contain ALL N paragraphs with word-level masking (`mask_text`). Paragraph dropout (`get_paragraph_mask`) applies only at the pooling stage, not at encoding. Passing a dropped subset to `masked_batch` causes L_local shape mismatch.

### GlocalIB Four-component loss with Homoscedastic Uncertainty Weighting

```
L = (L_compress · exp(−s₀) + s₀)   # KL( N(μ,σ²) ∥ N(0,1) )
  + (L_local   · exp(−s₁) + s₁)   # cosine dist: all N student para reps vs teacher para reps
  + (L_inter   · exp(−s₂) + s₂)   # cosine dist: student M-para pool vs teacher full-doc (pre-IB)
  + (L_global  · exp(−s₃) + s₃)   # cosine dist: Z_proj vs Z_prime (post-IB)

log_s = nn.Parameter(torch.zeros(4))   # clamped to [-10, 10]
```

EMA update: `model.update_teacher_ema()` called after every `optimizer.step()`.

### H-MLM baseline (scripts/03_pretrain_mlm.py)

Hierarchical MLM inspired by SMITH (Yang et al. ACL 2020). Processes all N paragraphs separately (not truncated concatenation). Two losses, weighted 0.5/0.5:

- **L_mlm**: Standard 15% masked token prediction per paragraph via `DataCollatorForLanguageModeling`
- **L_para_pred**: `alignment_loss(attn_pool(M kept para reps), stop-grad mean of all N para reps)` — trains `AttentionPooling` to reconstruct the full document from a partial view

Jointly pre-trains encoder + `AttentionPooling`. `gradient_checkpointing_enable()` is called on the `RobertaForMaskedLM` model (matching `GlocalIBModel`). At fine-tune time, both the encoder and the pooling module are loaded from the H-MLM checkpoint (unlike vanilla MLM which would leave the attention pool randomly initialized).

### Data flow through training

`mask_text(paragraph)` → word/sentence masked string (Xm per paragraph).
`get_paragraph_mask(n)` → kept indices list (30% dropped) — paragraph dropout for pooling stage only.
`load_ecthr()` filters documents: 5 ≤ paragraphs ≤ 50.
`sample_few_shot(split, n_per_class, seed)` is multi-label aware — deduplicates.

### Checkpoint formats

**GlocalIB** (`scripts/02_pretrain_glocal.py`) saves two files per epoch:
- `checkpoints/{CONDITION}_epoch{N}/` — full accelerator state (resumable)
- `checkpoints/{CONDITION}_epoch{N}.pt` — plain `model.state_dict()` (used by fine-tuning)

**H-MLM** (`scripts/03_pretrain_mlm.py`) saves per epoch:
- `checkpoints/h_mlm_epoch{N}.pt` — dict `{"encoder_state": roberta_state, "attn_pool_state": attn_pool_state}`

Fine-tuning (`scripts/04_finetune.py`) loads H-MLM encoder with `RobertaModel.from_pretrained("distilroberta-base", add_pooling_layer=False)` because `RobertaForMaskedLM` saves the encoder without the pooler layer. `load_encoder()` is called once per condition; the returned weights are snapshotted and restored before every `run_few_shot` call so all 15 seed×N runs start from identical pre-trained weights.

**`archive/`** contains the original root-level `train_glocal.py` and `train_mlm.py`. These use a deprecated API (`mask_paragraphs`, old `glocal_ib_loss` signatures) and are incompatible with the current checkpoints. Do not use them.

## src/ API

**`src/data.py`**
- `load_ecthr(min_paragraphs=5, max_paragraphs=50)` → `DatasetDict`
- `mask_text(paragraph, sent_dropout=0.20, span_rate=0.15)` → `str` — word/sentence masking within a paragraph
- `get_paragraph_mask(n_paragraphs, dropout_rate=0.30)` → `list[int]` — paragraph-level dropout indices
- `mask_paragraphs(paragraphs)` → `(list[str], list[int])` — **legacy, do not use with v3 training**: drops paragraphs before encoding, which breaks L_local alignment
- `sample_few_shot(split, n_per_class, seed, num_classes=10)` → `list[dict]`

**`src/model.py`**
- `GlocalIBModel(hidden_dim=256, proj_dim=512, max_paragraphs=50, ema_tau=0.99, device="cuda")`
  - forward: `(full_batch, masked_batch, kept_indices_batch)` → 8-tuple `(Z_prime, Z_proj, z_partial, s_chunks, t_chunks, mu, sigma, log_s)`
  - `update_teacher_ema()` — call after every optimizer step
- `DocumentClassifier(encoder, tokenizer, attn_pool, num_labels=10, max_paragraphs=50, device="cuda")` — forward: `(paragraphs_batch)` → `(B, 10)` sigmoid; uses CLS token, not pooler output
- `AttentionPooling(dim=768, max_chunks=50)` — learnable query + positional embeddings, zero-init → starts as mean pooling

**`src/loss.py`**
- `glocal_ib_loss(Z_prime, Z_proj, z_partial, s_chunks, t_chunks, mu, sigma, log_s)` → `(total, l_compress, l_local, l_inter, l_global)`
- `alignment_loss(z1, z2)` — `1 − mean cosine similarity`, expects `(N, D)` tensors
- `compression_loss(mu, sigma)` — KL divergence closed form

## Experimental Conditions

| Condition | Pre-training | Key difference |
|-----------|-------------|----------------|
| `glocal_ib` | 4-loss IB + UW + EMA attention | IB bottleneck forces compression |
| `mlm` (H-MLM) | Per-paragraph MLM + paragraph prediction | No IB; pre-trains same attention pool |

Fine-tuning: N ∈ {10, 50, 100} × 5 seeds. Results saved to `results/finetuning_results.json` as `{condition: {N: {"macro_f1": [...], "micro_f1": [...]}}}`. **Primary metric: macro-F1** (class imbalance is severe: label 3 has 4704 training examples, label 5 has 41). Micro-F1 reported as secondary metric.

## Key Invariants

- `distilroberta-base` loads via `RobertaModel.from_pretrained("distilroberta-base")` — no `DistilRobertaModel`.
- Teacher encoder pass is always `stop_grad=True` (`torch.no_grad()` inside `_encode_paragraphs`).
- Teacher/student encoding passes are sequential — never simultaneous — to minimize peak VRAM.
- `log_s` clamped to `[-10, 10]` in every forward pass (prevents numerical explosion).
- `update_teacher_ema()` must be called after every `optimizer.step()` — not inside `forward`.
- Both pre-training scripts use `get_linear_schedule_with_warmup` with 10% warmup steps over total training steps. GlocalIB LR=1e-5, H-MLM LR=5e-5. The schedulers are `accelerator.prepare()`d alongside the optimizer.
- If all four `log_s` stay near 0 through epoch 2, UW is degenerate — flag it, don't ignore.
- `scripts/` contains standalone Python equivalents of all notebooks (SSH/cluster friendly). Notebooks in `notebooks/` are kept for interactive use.

## Compute

| Job | Recommended | Batch config |
|-----|------------|-------------|
| GlocalIB pre-training | 4× A100 (Spark) | `BATCH_SIZE=1`, `GRAD_ACCUM=4` → effective 16 |
| H-MLM pre-training | 4× A100 (Spark) | `BATCH_SIZE=1`, `GRAD_ACCUM=4` (paragraph-by-paragraph) |
| Fine-tuning / exploration | 1× 32GB GPU | single example per step |
| Sanity checks | CPU | no GPU required |

On 10× 11GB GPUs: keep `BATCH_SIZE=1`, `GRAD_ACCUM=4` → effective batch 40. W&B project: `glocal-nlp`.
