# Final Architecture — GlocalDoc & H-MLM

Canonical reference for the two pre-training conditions evaluated in this project. Trust this document and the contents of `src/` over `doc/PLAN.md` (stale snapshot, 2026-05-07).

---

## 1. Overview

Both conditions pre-train `distilroberta-base` on the ECtHR Article-A judgments in LexGLUE, then fine-tune the resulting encoder plus a document pooler on few-shot multi-label classification. They share the same backbone, the same paragraph-by-paragraph processing pipeline, the same `AttentionPooling` architecture, and the same fine-tuning code. They differ in the **pre-training objective** only.

| | **GlocalDoc** (`glocal_ib`) | **H-MLM** (`h_mlm`) |
|---|---|---|
| Backbone | `distilroberta-base` (6 layers, 768-d, ~82M) | `distilroberta-base` |
| Architecture topology | Two-branch teacher + student | Single network |
| Masking strategy | Word/sentence masking (student) + paragraph dropout (pooling) | Token-level masking (15%, `DataCollatorForLanguageModeling`) for MLM head; unmasked paragraphs for pooling target |
| Attention pools | Two (`attn_pool_student`, `attn_pool_teacher`) | One (`attn_pool`) |
| Pre-training losses | 4 (`L_compress`, `L_local`, `L_inter`, `L_global`) | 2 (`L_mlm`, `L_para_pred`) |
| Loss balancing | Homoscedastic UW, 4 learnable `log_s` | Fixed `0.5 / 0.5` |
| Stop-grad target | EMA teacher pool over clean docs | Stop-grad mean of all-N paragraph reps |
| IB bottleneck | Yes (Gaussian, 256-d) | No |
| Encoder passes per doc | 2 (teacher / student) | 3 (MLM-masked, target no-grad, kept-paragraph grad) |
| LR / scheduler | `1e-5`, linear + 10% warmup | `5e-5`, linear + 10% warmup |
| Fine-tune loads | `encoder` + `attn_pool_teacher` | `encoder` + `attn_pool` |

---

## 2. Shared infrastructure

### 2.1 Backbone

```python
RobertaModel.from_pretrained("distilroberta-base")          # GlocalDoc
RobertaForMaskedLM.from_pretrained("distilroberta-base")    # H-MLM (adds LM head)
# There is no DistilRobertaModel class — distilroberta = a 6-layer RoBERTa.
encoder.gradient_checkpointing_enable()
```

### 2.2 Paragraph encoding (`_encode_paragraphs`)

Documents are kept as lists of paragraphs — never truncated to a single 512-token window. Per paragraph:

- ≤ 510 tokens → encoded directly; the `[CLS]` vector from `last_hidden_state[:, 0, :]` is taken.
- &gt; 510 tokens → split into non-overlapping 510-token sub-chunks, each encoded separately, then **mean-pooled** into one 768-d vector.

All sub-chunks for a document are concatenated and passed to the encoder in a **single forward pass**, so peak activation memory equals one pass regardless of paragraph count. `stop_grad=True` wraps the forward in `torch.no_grad()` (no activations retained).

### 2.3 `AttentionPooling`

```python
class AttentionPooling(nn.Module):
    def __init__(self, dim=768, max_chunks=50):
        self.attn_query = nn.Parameter(torch.randn(dim) * 0.01)
        self.chunk_pos  = nn.Embedding(max_chunks, dim)
        nn.init.zeros_(self.chunk_pos.weight)      # zero-init → starts as mean pooling
```

A learnable query attends over `(paragraph_rep + position_embedding)`; softmax weights produce a single 768-d document vector. With zero-initialised position embeddings the module starts as plain mean pooling and learns away from it.

### 2.4 Data API (`src/data.py`)

```python
load_ecthr(min_paragraphs=5, max_paragraphs=50)
# Filter: 5 ≤ paragraphs ≤ 50. Discards <1% of ECtHR (outliers + very short docs).

mask_text(paragraph, sent_dropout=0.20, span_rate=0.15)
# Stage 1: split on . ! ? — drop each sentence with p=0.20.
# Stage 2: span-mask 15% of remaining words with spans of length 3–5.
# Consecutive masks collapse to one <mask>.

get_paragraph_mask(n_paragraphs, dropout_rate=0.30)
# Returns sorted list of M = max(1, round(n · 0.7)) kept indices.

sample_few_shot(split, n_per_class, seed, num_classes=10)
# Multi-label aware: deduplicates docs that satisfy multiple classes.
```

`mask_paragraphs(paragraphs)` exists as a legacy API. **Do not use** in v3 training — it drops paragraphs *before* encoding, which breaks `L_local`'s shape alignment.

---

## 3. GlocalDoc (`glocal_ib`)

### 3.1 Data flow

```
clean paragraphs (N)
   └─► encoder (stop-grad) ─────────► t_chunks (N, 768)
                                          └─► attn_pool_teacher (EMA) ──► Z_prime (768)

masked paragraphs (N · mask_text)
   └─► encoder (trainable) ─────────► s_chunks (N, 768)
        │                                 ▲
        │   paragraph dropout via I       │
        ▼                                 │
   s_kept (M, 768)                        │
        └─► attn_pool_student ──► z_partial (768)
                                       ├─► mu_head        ──► μ          (256)
                                       └─► log_sigma_head ──► log σ      (256, clamped to [-10, 10])
                                                 │
                                                 ▼
                                   z = μ + exp(log σ) ⊙ ε,  ε ~ N(0, I)
                                                 │
                                                 ▼
                                          projector (256 → 512 → ReLU → 768) ──► Z_proj (768)
```

### 3.2 `GlocalIBModel` composition

| Member | Shape / type | Train mode |
|---|---|---|
| `encoder` | `RobertaModel` (distilroberta-base) | Trainable; teacher pass under `torch.no_grad()` |
| `attn_pool_student` | `AttentionPooling(768, 50)` | Trainable (gradient) |
| `attn_pool_teacher` | `AttentionPooling(768, 50)` | Initialised from `attn_pool_student.state_dict()`; **all params `requires_grad=False`**; updated by EMA (τ=0.99) |
| `mu_head` | `nn.Linear(768, 256)` | Trainable |
| `log_sigma_head` | `nn.Linear(768, 256)` | Trainable; **output clamped to `[-10, 10]`** for bf16 stability |
| `projector` | `Linear(256, 512) → ReLU → Linear(512, 768)` | Trainable |
| `log_s` | `nn.Parameter(torch.zeros(4))` | Trainable; **clamped to `[-10, 10]`** every forward |

### 3.3 Forward signature

```python
model.forward(full_batch, masked_batch, kept_indices_batch) → 8-tuple
# full_batch:         list[list[str]]    clean paragraphs (X0)
# masked_batch:       list[list[str]]    word/sentence-masked (Xm) — must contain ALL N
# kept_indices_batch: list[list[int]]    M kept indices per doc (paragraph dropout)
#
# returns:
#   Z_prime    (B, 768)        teacher full-doc repr (stop-grad)
#   Z_proj     (B, 768)        student post-IB projection
#   z_partial  (B, 768)        student pre-IB partial-doc pool (for L_inter)
#   s_chunks   list[(N, 768)]  student all-paragraph reps  (for L_local)
#   t_chunks   list[(N, 768)]  teacher all-paragraph reps  (for L_local)
#   mu         (B, 256)        IB mean
#   log_sigma  (B, 256)        IB log-std (clamped)
#   log_s      (4,)            UW weights (clamped)
```

**Two non-negotiable invariants:**

1. `masked_batch` carries **all N** paragraphs (word-masked). Paragraph dropout happens *after* encoding, by indexing `s_chunks[valid_kept]`. Passing an already-dropped subset to the student encoder breaks `L_local`'s shape alignment with `t_chunks`.
2. Teacher and student encoder passes are **sequential, never simultaneous**. Peak VRAM = one student-pass equivalent.

### 3.4 Loss — four components + homoscedastic UW

```python
# src/loss.py
l_compress = -0.5 · Σ_dim ( 1 + 2·log σ − μ² − exp(2·log σ) )           # bf16-stable KL
l_local    = 1 − mean_cosine( cat(s_chunks), cat(t_chunks) )            # all (B·N) pairs
l_inter    = 1 − mean_cosine( z_partial, Z_prime.detach() )
l_global   = 1 − mean_cosine( Z_proj,    Z_prime.detach() )

losses = stack([l_compress, l_local, l_inter, l_global])
total  = Σᵢ ( Lᵢ · exp(-sᵢ) + sᵢ )       # log_s clamped to [-10, 10]
```

Both `L_inter` and `L_global` anchor on the **same** `Z_prime` (one teacher full-doc rep), `.detach()`-ed so no gradient flows into the teacher pool through these terms — only through the EMA path. `L_local` operates on concatenated `(B·N, 768)` tensors, so cosine is element-wise per paragraph pair `(student_i, teacher_i)`.

### 3.5 EMA update

```python
def update_teacher_ema(self):
    with torch.no_grad():
        for p_t, p_s in zip(attn_pool_teacher.parameters(),
                            attn_pool_student.parameters()):
            p_t.data = τ · p_t.data + (1 − τ) · p_s.data       # τ = 0.99
```

Called **after every `optimizer.step()`**. The encoder is **not** EMA'd — it is gradient-shared between branches; the teacher just uses `torch.no_grad()`.

### 3.6 Training loop sketch (`scripts/02_pretrain_glocal.py`)

```python
accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=4)
opt   = AdamW(model.parameters(), lr=1e-5)
sched = get_linear_schedule_with_warmup(opt, int(0.1·total), total)
model, opt, sched, loader = accelerator.prepare(model, opt, sched, loader)

for full_batch in loader:
    masked_batch       = [[mask_text(p) for p in doc] for doc in full_batch]
    kept_indices_batch = [get_paragraph_mask(len(doc)) for doc in full_batch]

    with accelerator.accumulate(model):
        out = model(full_batch, masked_batch, kept_indices_batch)
        total, l_c, l_l, l_i, l_g = glocal_ib_loss(*out)
        accelerator.backward(total)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad()
        if accelerator.sync_gradients:
            accelerator.unwrap_model(model).update_teacher_ema()
```

W&B keys: `total_loss`, `l_compress`, `l_local`, `l_inter`, `l_global`, `log_s_{compress,local,inter,global}`, `uw_contrib_{compress,local,inter,global}` (the raw loss × `exp(-s)` multiplier — the diagnostic for whether UW has gone degenerate).

### 3.7 Checkpoints

Per epoch:
- `checkpoints/glocal_ib_epoch{N}/` — full accelerator state (resumable)
- `checkpoints/glocal_ib_epoch{N}.pt` — plain `model.state_dict()` (used by fine-tuning)

---

## 4. H-MLM (`h_mlm`) — inspired by SMITH (Yang et al., CIKM 2020)

### 4.1 Data flow

```
all N paragraphs                              all N paragraphs                            M kept paragraphs (indices I)
   (token-masked 15%)                          (unmasked)                                  (unmasked, grad path)
        │                                            │                                            │
        ▼                                            ▼                                            ▼
 RobertaForMaskedLM.forward                roberta(...)  under no_grad             roberta(...)  with grad
        │                                            │                                            │
        ▼                                            ▼                                            ▼
   token logits + labels                      per-paragraph [CLS]                       per-paragraph [CLS]
        │                                       (N, 768)                                       (M, 768)
        ▼                                            │                                            │
       L_mlm                                         ▼                                            ▼
                                            mean ──► doc_target (768)                  attn_pool ──► doc_partial (768)
                                                          │                                          │
                                                          └─── L_para_pred = 1 − cos(doc_partial, doc_target.detach())
```

Three encoder forward passes per document, kept separate for gradient-flow efficiency: a no-grad pass on all N gives the target representation cheaply; a separate with-grad pass on the M kept paragraphs supplies the gradient signal to the encoder *through the attention pool*.

### 4.2 Composition (`HMLMTrainer` inside `scripts/03_pretrain_mlm.py`)

| Member | Shape / type | Train mode |
|---|---|---|
| `roberta_for_mlm` | `RobertaForMaskedLM.from_pretrained("distilroberta-base")` | Trainable; `gradient_checkpointing_enable()` |
| `attn_pool` | `AttentionPooling(768, 50)` | Trainable |

Both members are bundled into a single `nn.Module` and passed **once** to `accelerator.prepare()`. This is load-bearing: DDP all-reduces every parameter's gradients, and mixed-precision autocast covers every encoder call. **Never** call `accelerator.unwrap_model(...)` for a forward — only for saving state.

### 4.3 Forward (per document)

1. Build masked input via `DataCollatorForLanguageModeling(mlm_probability=0.15)` → token-level masked input + labels.
2. `roberta_for_mlm(**masked_input).loss` → `L_mlm` (the LM head's cross-entropy over masked tokens).
3. `with torch.no_grad(): t_cls = roberta(unmasked_all_N).last_hidden_state[:, 0, :]` → `(N, 768)` paragraph CLSs, stop-grad. Take their **mean** → `doc_target (768)`.
4. `kept = get_paragraph_mask(N)` → M indices.
5. `s_cls = roberta(unmasked_kept_M).last_hidden_state[:, 0, :]` → `(M, 768)`, with gradient.
6. `doc_partial = attn_pool(s_cls)` → `(768,)`.
7. `L_para_pred = alignment_loss(doc_partial.unsqueeze(0), doc_target.detach().unsqueeze(0))` → `1 − cos`.

### 4.4 Loss

```python
L = 0.5 · L_mlm  +  0.5 · L_para_pred
```

Fixed weighting; no learnable scales. The 0.5 / 0.5 split is by design, not tuning.

### 4.5 Training loop sketch (`scripts/03_pretrain_mlm.py`)

```python
accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=4)
opt   = AdamW(trainer.parameters(), lr=5e-5)
sched = get_linear_schedule_with_warmup(opt, int(0.1·total), total)
trainer, opt, sched, loader = accelerator.prepare(trainer, opt, sched, loader)

for batch in loader:
    with accelerator.accumulate(trainer):
        loss, l_mlm, l_pp = trainer(batch)   # forward computes both, returns the weighted sum
        accelerator.backward(loss)
        opt.step(); sched.step(); opt.zero_grad()
```

W&B keys: `total`, `l_mlm`, `l_para_pred`.

### 4.6 Checkpoint

Per epoch:
- `checkpoints/h_mlm_epoch{N}.pt` — `{"encoder_state": roberta_state, "attn_pool_state": attn_pool_state}`

Fine-tuning loads the encoder with:
```python
RobertaModel.from_pretrained("distilroberta-base", add_pooling_layer=False)
# RobertaForMaskedLM saves the encoder without the pooler layer; this matches the saved shape.
```

---

## 5. Fine-tuning (`DocumentClassifier`, `scripts/04_finetune.py`)

Shared between both conditions.

```python
DocumentClassifier(encoder, tokenizer, attn_pool, num_labels=10, max_paragraphs=50)
# forward(paragraphs_batch: list[list[str]]) → (B, 10) raw logits
# Pair with BCEWithLogitsLoss at train; apply sigmoid at eval. Never double-sigmoid.
```

### 5.1 What gets loaded per condition

- **GlocalDoc** → `encoder = pretrained.encoder`, `attn_pool = pretrained.attn_pool_teacher`. The EMA-stabilised teacher pool is the right choice: it saw full documents (no paragraph dropout) throughout pre-training, which matches the no-dropout fine-tune setting.
- **H-MLM** → `encoder` from `checkpoint["encoder_state"]`, `attn_pool` from `checkpoint["attn_pool_state"]`.

### 5.2 Protocol

```python
# 5 seeds × 3 budgets (N ∈ {10, 50, 100}) × 2 conditions = 30 runs total.
# load_encoder(condition) → called once per condition; weights snapshotted.

for seed in seeds:
    for n_per_class in [10, 50, 100]:
        seed_everything(seed)                   # BEFORE sample AND BEFORE classifier construction
        restore_weights_from_snapshot()
        train_sample = sample_few_shot(train, n_per_class, seed)
        clf          = DocumentClassifier(encoder, tokenizer, attn_pool, ...)
        ...
```

If `seed_everything` is called *after* the classifier is constructed, the head's `nn.Linear` init does not depend on the seed and the five seeds collapse to only varying the few-shot sample.

### 5.3 Output

```
results/finetuning_results.json
{
  "glocal_ib": {"10": {"macro_f1": [s1..s5], "micro_f1": [s1..s5]}, ...},
  "h_mlm":     {"10": {"macro_f1": [...],    "micro_f1": [...]}, ...}
}
```

**Primary metric: macro-F1.** The ten ECtHR violation labels span a ~100× frequency gap (label 3: 4704 train examples; label 5: 41). Micro-F1 is reported as a secondary metric.

---

## 6. Key invariants

- `distilroberta-base` loads via `RobertaModel.from_pretrained("distilroberta-base")` — there is no `DistilRobertaModel`.
- Teacher encoder pass is always `stop_grad=True`. Encoder is not EMA'd; only the teacher attention pool is.
- `log_s` and `log_sigma` are both clamped to `[-10, 10]` in every forward pass.
- `update_teacher_ema()` is called **after every `optimizer.step()`** — not inside `forward`.
- GlocalDoc `masked_batch` carries all N paragraphs (word-masked). Paragraph dropout applies at the pooling stage by indexing `s_chunks[valid_kept]`, **not** at the encoder input.
- Both pre-training scripts use `get_linear_schedule_with_warmup` with 10% warmup over total training steps, prepared alongside the optimizer via `accelerator.prepare(...)`.
- In fine-tuning, `seed_everything(seed)` runs **before** `sample_few_shot` and **before** `DocumentClassifier(...)` construction. Otherwise head init is seed-independent and the five seeds collapse.
- SLURM scripts use `ntasks-per-node=1` because `accelerate launch --num_processes=N` spawns its own workers; `ntasks-per-node=N` would create N×N processes fighting over N GPUs.
- The `archive/` directory contains the original root-level `train_glocal.py` / `train_mlm.py`. They use a deprecated API and are incompatible with the current checkpoints. **Do not use.**

---

## 7. Conditions table — the experiment

| Condition | Pre-training objective | Loss components | Pool pre-trained? |
|---|---|---|---|
| `glocal_ib` | IB compression + multi-granularity teacher–student alignment | 4, balanced by UW | Yes (EMA teacher) |
| `h_mlm` | Token MLM + partial-pool ↔ full-doc-mean alignment | 2, fixed 0.5 / 0.5 | Yes (single pool) |

Both conditions reach fine-tuning with the **same architecture** (encoder + attention pool), and the same data. The single variable being tested is the **pre-training objective**.

Fine-tuning evaluation: `N ∈ {10, 50, 100}` × 5 seeds per condition. Primary metric: **macro-F1**.
