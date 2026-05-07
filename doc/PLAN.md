# GlocalDoc v3 — Implementation Plan

> **Self-contained.** This document can be followed from a fresh clone with no prior context.
> Supersedes all earlier plans. Last updated 2026-05-07.

---

## Background

**Project:** MSc research at Free University of Bozen-Bolzano.
Adapts the GlocalIB (Global-Local Information Bottleneck) objective from time series imputation to
few-shot legal document classification on the ECtHR dataset (LexGLUE / `coastalcph/lex_glue`).

**v3 redesign rationale:**
v2 masked entire paragraphs from the student input, which forced all compression to happen at
the paragraph-count level. v3 adds word/sentence masking within paragraphs (X0 → Xm), producing
a richer two-level compression signal: the student must handle both corrupted text and missing
paragraphs. The attention pooling is redesigned with separate teacher/student modules and
BYOL-style EMA updates, resolving a training-distribution mismatch in v2.

---

## Repository Layout

```
GlocalDoc/
├── src/
│   ├── __init__.py
│   ├── data.py          ← MODIFIED (new masking functions + updated filters)
│   ├── model.py         ← FULL REWRITE
│   └── loss.py          ← UPDATED
├── scripts/
│   ├── 01_data_exploration.py
│   ├── 02_pretrain_glocal.py
│   ├── 03_pretrain_mlm.py
│   └── 04_finetune.py
├── notebooks/           ← interactive use only
├── slurm/
│   ├── pretrain_glocal.sh
│   ├── pretrain_mlm.sh
│   └── finetune.sh
├── checkpoints/         ← gitignored
├── results/             ← gitignored
├── logs/                ← gitignored
├── doc/
│   ├── PLAN.md          ← this file
│   └── TODO.md
└── requirements.txt
```

---

## Environment Setup (do once)

```bash
conda create -n glocal_nlp python=3.10
conda activate glocal_nlp
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt

# Verify
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

**`requirements.txt`:**
```
torch
transformers
datasets
scikit-learn
wandb
jupyter
nbconvert
accelerate
```

---

## Single Machine vs GPU Cluster

No code change between environments. `accelerate` handles it via config.

```python
accelerator = Accelerator(
    mixed_precision="bf16",
    gradient_accumulation_steps=4,
)
```

| Scenario | How to run |
|----------|-----------|
| Single GPU | `python scripts/02_pretrain_glocal.py` |
| CPU debug | same (bf16 gracefully ignored) |
| 4 GPUs DDP | `accelerate launch --num_processes=4 scripts/02_pretrain_glocal.py` |
| SLURM | see `slurm/` scripts |

---

## Architecture

### Overview: Two Masking Levels

```
X0 (clean text)  ──► Teacher Encoder (stop-grad) ──► N paragraph reps (768)
                                                              │
                                              ┌───────────────────────────────┐
                                              │  L_local: align all N pairs   │
                                              └───────────────────────────────┘
Xm (masked text) ──► Student Encoder (trainable) ──► N paragraph reps (768)
                                                              │
                                                   drop 30% → M reps
                                                              │
                                               attn_pool_student (trainable)
                                                              │
                                                        z_partial (768)
                                                              │
          L_inter ◄──── cosine_dist ─────────────────────────┤──────────────► Z_prime
                                                              │                    ▲
                                                           IB head                │
                                                         (mu, sigma)        attn_pool_teacher (EMA)
                                                              │                    │
                                                         Z_proj (768)      N teacher reps
                                                              │
          L_global ◄──── cosine_dist ──────────────────────────────────────► Z_prime
```

**Masking level 1 (text):** word/sentence masking within each paragraph → student sees degraded
text but all N paragraphs.
**Masking level 2 (paragraphs):** 30% of encoded student paragraph vectors dropped before
attention pooling → student aggregates from partial evidence.

---

### 1. Encoder: `distilroberta-base`

```
HuggingFace ID:  "distilroberta-base"
Load as:         RobertaModel.from_pretrained("distilroberta-base")
                 (there is no DistilRobertaModel class — distilroberta is a 6-layer RoBERTa)
Parameters:      ~82M
Hidden dim:      768
Tokenizer:       RobertaTokenizerFast
```

Gradient checkpointing enabled: `encoder.gradient_checkpointing_enable()`

Teacher branch: stop-gradient via `torch.no_grad()`. Student branch: full gradient tracking.
These two encoder passes are always **sequential, never simultaneous** — peak activation VRAM
equals one forward pass over N paragraphs, not two.

---

### 2. Data Filtering

```python
# In load_ecthr():
dataset = dataset.filter(lambda x: 5 <= len(x["text"]) <= 50)
```

Documents with fewer than 5 paragraphs cannot be masked meaningfully.
Documents with more than 50 paragraphs are excluded entirely (outliers: <1% of ECtHR).
No symmetric truncation — clean filter only.

---

### 3. Word/Sentence Masking (`mask_text`)

Applied to each paragraph before student encoding. Two stages, applied sequentially:

**Stage 1 — Sentence dropout (20%):**
Split paragraph into sentences on `.`, `?`, `!`. Drop each sentence independently with p=0.20.
Dropped sentences are removed (not replaced). If all sentences are dropped, return original paragraph.

**Stage 2 — Span masking (15% of words, spans of 3–5):**
On remaining text after sentence dropout, randomly sample word-level spans of 3–5 words
until ~15% of words are covered. Replace each span with a single `<mask>` token.
Consecutive `<mask>` tokens are collapsed to one.

`<mask>` is the RoBERTa mask token and is handled correctly by `RobertaTokenizerFast`.

---

### 4. Paragraph Dropout (`get_paragraph_mask`)

Applied at the pooling stage — not at encoding. The student encodes all N paragraphs; only
the pooling step uses a subset.

```python
def get_paragraph_mask(n_paragraphs: int, dropout_rate: float = 0.30) -> list[int]:
    n_keep = max(1, round(n_paragraphs * (1 - dropout_rate)))
    return sorted(random.sample(range(n_paragraphs), n_keep))
```

Returns M kept indices. Rate: 30% dropped, fixed (no curriculum for now).

---

### 5. Long Paragraph Handling

Paragraphs with more than 510 tokens (leaving room for `[CLS]` and `[SEP]`) are split into
non-overlapping sub-chunks of 510 tokens, encoded separately, and mean-pooled into one 768-dim
vector. This is handled inside `_encode_paragraphs()` and is transparent to all callers.

Both `GlocalIBModel` and `DocumentClassifier` use identical paragraph encoding. Consistency
between pre-training and fine-tuning is required for representations to transfer correctly.

```python
def _encode_paragraphs(self, paragraphs: list[str], stop_grad: bool) -> torch.Tensor:
    CHUNK_SIZE = 510
    sub_chunks, boundaries = [], []

    for para in paragraphs:
        ids = self.tokenizer.encode(para, add_special_tokens=False)
        start = len(sub_chunks)
        if len(ids) <= CHUNK_SIZE:
            sub_chunks.append(para)
        else:
            for i in range(0, len(ids), CHUNK_SIZE):
                sub_chunks.append(self.tokenizer.decode(ids[i:i + CHUNK_SIZE]))
        boundaries.append((start, len(sub_chunks)))

    enc = self.tokenizer(
        sub_chunks, padding=True, truncation=True,
        max_length=512, return_tensors="pt",
    ).to(self.encoder.device)

    ctx = torch.no_grad() if stop_grad else contextlib.nullcontext()
    with ctx:
        cls_vecs = self.encoder(**enc).last_hidden_state[:, 0, :]  # (total_chunks, 768)

    return torch.stack([cls_vecs[s:e].mean(0) for s, e in boundaries])  # (N, 768)
```

---

### 6. Attention Pooling — Two Modules + EMA

```python
class AttentionPooling(nn.Module):
    def __init__(self, dim: int = 768, max_chunks: int = 50):
        super().__init__()
        self.attn_query = nn.Parameter(torch.randn(dim) * 0.01)
        self.chunk_pos  = nn.Embedding(max_chunks, dim)
        nn.init.zeros_(self.chunk_pos.weight)   # zero-init → starts as mean pooling

    def forward(self, chunk_vecs: torch.Tensor, return_weights: bool = False):
        N       = chunk_vecs.size(0)
        pos_ids = torch.arange(N, device=chunk_vecs.device)
        vecs    = chunk_vecs + self.chunk_pos(pos_ids)
        scores  = vecs @ self.attn_query                    # (N,)
        weights = torch.softmax(scores, dim=0)              # (N,)
        doc_vec = (weights.unsqueeze(-1) * vecs).sum(0)     # (768,)
        if return_weights:
            return doc_vec, weights
        return doc_vec
```

Two separate instances in `GlocalIBModel`:

| Module | Trained by | Role |
|--------|-----------|------|
| `attn_pool_student` | Gradient descent | Aggregates M student paragraph reps |
| `attn_pool_teacher` | EMA of student (τ=0.99) | Aggregates N teacher paragraph reps; used for fine-tuning |

**EMA update** (called after every `optimizer.step()`):
```python
def _update_teacher_ema(self, tau: float = 0.99):
    with torch.no_grad():
        for p_t, p_s in zip(
            self.attn_pool_teacher.parameters(),
            self.attn_pool_student.parameters()
        ):
            p_t.data = tau * p_t.data + (1 - tau) * p_s.data
```

Teacher attention is always wrapped in `torch.no_grad()` for loss computation — no gradients
flow through it. The EMA keeps teacher as a slowly-evolving stable target.

`DocumentClassifier` uses `attn_pool_teacher` at fine-tuning because the teacher attention
was shaped by full-document inputs (all N paragraphs) throughout pre-training, matching the
fine-tuning scenario where no paragraph dropout is applied.

---

### 7. Four-Component Loss + Homoscedastic Uncertainty Weighting

```
L_total = Σᵢ [ Lᵢ · exp(−sᵢ) + sᵢ ]

  L_compress = KL( N(μ,σ²) ∥ N(0,1) )
             = −0.5 · Σ(1 + 2·log σ − μ² − σ²)

  L_local    = cosine_dist(s_chunks_all_N, t_chunks_all_N)
               (all N paragraph pairs — both branches encode all N paragraphs)

  L_inter    = cosine_dist(z_partial, Z_prime)
               (pre-IB: student M-paragraph pool vs teacher full-doc pool)
               (direct gradient to student attention, bypasses IB bottleneck)

  L_global   = cosine_dist(Z_proj, Z_prime)
               (post-IB: student IB projection vs teacher full-doc pool)

  log_s = nn.Parameter(torch.zeros(4))          # [compress, local, inter, global]
  log_s = torch.clamp(log_s, min=-10, max=10)   # prevents numerical explosion
```

**Why UW works:** increasing sᵢ reduces the task weight `exp(−sᵢ)` but raises the regularization
cost `sᵢ` at the same rate. These forces balance at `sᵢ = log(Lᵢ)`, automatically normalizing
all losses by their magnitude. No manual weight tuning needed.

**Diagnostic:** log all four `sᵢ` values to W&B every step. If all stay near 0 by epoch 2,
the UW is degenerate — escalate and investigate.

---

### 8. IB Probabilistic Head + Projector

```
z_partial (768) → mu_head    → μ (256)
z_partial (768) → log_σ_head → log_σ → σ = exp(log_σ)
Z_sample = μ + σ · ε,   ε ~ N(0, I)    [reparameterization]
Z_proj = projector(Z_sample)            256 → 512 → ReLU → 768
```

```python
self.mu_head        = nn.Linear(768, 256)
self.log_sigma_head = nn.Linear(768, 256)
self.projector      = nn.Sequential(
    nn.Linear(256, 512),
    nn.ReLU(),
    nn.Linear(512, 768),
)
```

---

## File Implementations

### `src/data.py` — Modified

Add two functions. Keep existing `load_ecthr`, `mask_paragraphs`, `sample_few_shot` signatures
but update `load_ecthr` filter.

```python
import re
import random
from datasets import load_dataset, DatasetDict


def load_ecthr(min_paragraphs: int = 5, max_paragraphs: int = 50) -> DatasetDict:
    dataset = load_dataset("coastalcph/lex_glue", "ecthr_a")
    return dataset.filter(
        lambda x: min_paragraphs <= len(x["text"]) <= max_paragraphs
    )


def mask_text(
    paragraph: str,
    sent_dropout: float = 0.20,
    span_rate: float = 0.15,
    span_len_min: int = 3,
    span_len_max: int = 5,
) -> str:
    """Two-stage masking: sentence dropout then span masking."""
    # Stage 1: sentence dropout
    sentences = re.split(r'(?<=[.!?])\s+', paragraph.strip())
    kept = [s for s in sentences if random.random() > sent_dropout]
    if not kept:
        kept = sentences  # fallback: keep all if all dropped
    text = ' '.join(kept)

    # Stage 2: span masking at word level
    words = text.split()
    if not words:
        return paragraph
    n_to_mask = max(1, round(len(words) * span_rate))
    masked = set()
    attempts = 0
    while len(masked) < n_to_mask and attempts < len(words) * 3:
        attempts += 1
        start = random.randint(0, len(words) - 1)
        span  = random.randint(span_len_min, span_len_max)
        for j in range(start, min(start + span, len(words))):
            masked.add(j)

    result, prev_mask = [], False
    for i, w in enumerate(words):
        if i in masked:
            if not prev_mask:
                result.append('<mask>')
            prev_mask = True
        else:
            result.append(w)
            prev_mask = False
    return ' '.join(result)


def get_paragraph_mask(n_paragraphs: int, dropout_rate: float = 0.30) -> list[int]:
    """Returns sorted list of kept paragraph indices (30% dropped)."""
    n_keep = max(1, round(n_paragraphs * (1 - dropout_rate)))
    return sorted(random.sample(range(n_paragraphs), n_keep))


def mask_paragraphs(paragraphs: list[str]) -> tuple[list[str], list[int]]:
    """Legacy API: paragraph-level dropout only. Returns (kept_paragraphs, kept_indices).
    Still used by scripts that reference the old API."""
    indices = get_paragraph_mask(len(paragraphs))
    return [paragraphs[i] for i in indices], indices


def sample_few_shot(split, n_per_class: int, seed: int, num_classes: int = 10) -> list[dict]:
    """Multi-label aware few-shot sampling. Deduplicates docs satisfying multiple classes."""
    rng = random.Random(seed)
    per_class = {c: [] for c in range(num_classes)}
    for ex in split:
        for label in ex["labels"]:
            per_class[label].append(ex)
    selected, seen_ids = [], set()
    for c in range(num_classes):
        pool = [ex for ex in per_class[c] if id(ex) not in seen_ids]
        rng.shuffle(pool)
        for ex in pool[:n_per_class]:
            selected.append(ex)
            seen_ids.add(id(ex))
    return selected
```

---

### `src/model.py` — Full Rewrite

```python
import contextlib
import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizerFast


class AttentionPooling(nn.Module):
    def __init__(self, dim: int = 768, max_chunks: int = 50):
        super().__init__()
        self.attn_query = nn.Parameter(torch.randn(dim) * 0.01)
        self.chunk_pos  = nn.Embedding(max_chunks, dim)
        nn.init.zeros_(self.chunk_pos.weight)

    def forward(self, chunk_vecs: torch.Tensor, return_weights: bool = False):
        N       = chunk_vecs.size(0)
        pos_ids = torch.arange(N, device=chunk_vecs.device)
        vecs    = chunk_vecs + self.chunk_pos(pos_ids)
        scores  = vecs @ self.attn_query
        weights = torch.softmax(scores, dim=0)
        doc_vec = (weights.unsqueeze(-1) * vecs).sum(0)
        if return_weights:
            return doc_vec, weights
        return doc_vec


class GlocalIBModel(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        proj_dim: int   = 512,
        max_paragraphs: int = 50,
        ema_tau: float  = 0.99,
        device: str     = "cuda",
    ):
        super().__init__()
        self.tokenizer = RobertaTokenizerFast.from_pretrained("distilroberta-base")
        self.encoder   = RobertaModel.from_pretrained("distilroberta-base")
        self.encoder.gradient_checkpointing_enable()

        self.attn_pool_student = AttentionPooling(dim=768, max_chunks=max_paragraphs)
        self.attn_pool_teacher = AttentionPooling(dim=768, max_chunks=max_paragraphs)

        self.mu_head        = nn.Linear(768, hidden_dim)
        self.log_sigma_head = nn.Linear(768, hidden_dim)
        self.projector      = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, 768),
        )

        self.log_s     = nn.Parameter(torch.zeros(4))   # UW weights [compress, local, inter, global]
        self.ema_tau   = ema_tau
        self.max_paragraphs = max_paragraphs
        self.to(device)

    def _encode_paragraphs(self, paragraphs: list[str], stop_grad: bool) -> torch.Tensor:
        """
        Returns (N, 768). Paragraphs >510 tokens are split into sub-chunks,
        encoded separately, and mean-pooled into one vector each.
        Single batched encoder forward pass over all sub-chunks.
        """
        CHUNK_SIZE = 510
        sub_chunks: list[str] = []
        boundaries: list[tuple[int, int]] = []

        for para in paragraphs:
            ids   = self.tokenizer.encode(para, add_special_tokens=False)
            start = len(sub_chunks)
            if len(ids) <= CHUNK_SIZE:
                sub_chunks.append(para)
            else:
                for i in range(0, len(ids), CHUNK_SIZE):
                    sub_chunks.append(self.tokenizer.decode(ids[i:i + CHUNK_SIZE]))
            boundaries.append((start, len(sub_chunks)))

        enc = self.tokenizer(
            sub_chunks, padding=True, truncation=True,
            max_length=512, return_tensors="pt",
        ).to(self.encoder.device)

        ctx = torch.no_grad() if stop_grad else contextlib.nullcontext()
        with ctx:
            cls_vecs = self.encoder(**enc).last_hidden_state[:, 0, :]  # (total_chunks, 768)

        return torch.stack([cls_vecs[s:e].mean(0) for s, e in boundaries])  # (N, 768)

    def update_teacher_ema(self):
        """Call after every optimizer.step(). Updates teacher attention via EMA."""
        with torch.no_grad():
            for p_t, p_s in zip(
                self.attn_pool_teacher.parameters(),
                self.attn_pool_student.parameters(),
            ):
                p_t.data = self.ema_tau * p_t.data + (1 - self.ema_tau) * p_s.data

    def forward(
        self,
        full_batch:         list[list[str]],   # clean paragraphs (X0)
        masked_batch:       list[list[str]],   # word/sentence masked (Xm)
        kept_indices_batch: list[list[int]],   # M kept indices per doc (paragraph dropout)
    ):
        """
        Returns 8-tuple:
          Z_prime   (B, 768)        teacher full-doc repr (stop-grad)
          Z_proj    (B, 768)        student post-IB projection
          z_partial (B, 768)        student pre-IB partial-doc repr (for L_inter)
          s_chunks  list[(N, 768)]  student all-paragraph reps (for L_local)
          t_chunks  list[(N, 768)]  teacher all-paragraph reps (for L_local)
          mu        (B, 256)        IB mean
          sigma     (B, 256)        IB std
          log_s     (4,)            UW weights (clamped)
        """
        Z_prime_list, Z_proj_list, z_partial_list = [], [], []
        s_chunks_list, t_chunks_list = [], []
        mu_list, sigma_list = [], []

        for full, masked, kept in zip(full_batch, masked_batch, kept_indices_batch):
            # ── Teacher pass (stop-grad, no activations stored) ──────────────
            t_chunks = self._encode_paragraphs(full, stop_grad=True)       # (N, 768)
            with torch.no_grad():
                Z_prime = self.attn_pool_teacher(t_chunks)                 # (768,)

            # ── Student pass (full grad) ──────────────────────────────────────
            s_chunks = self._encode_paragraphs(masked, stop_grad=False)    # (N, 768)

            # Paragraph dropout: keep only M reps for pooling
            valid_kept = [i for i in kept if i < len(s_chunks)]
            if not valid_kept:
                valid_kept = [0]
            s_kept    = s_chunks[valid_kept]                               # (M, 768)
            z_partial = self.attn_pool_student(s_kept)                    # (768,)

            # IB head
            mu       = self.mu_head(z_partial)
            sigma    = torch.exp(self.log_sigma_head(z_partial))
            z_sample = mu + sigma * torch.randn_like(mu)
            Z_proj   = self.projector(z_sample)                           # (768,)

            Z_prime_list.append(Z_prime)
            Z_proj_list.append(Z_proj)
            z_partial_list.append(z_partial)
            s_chunks_list.append(s_chunks)
            t_chunks_list.append(t_chunks)
            mu_list.append(mu)
            sigma_list.append(sigma)

        log_s = torch.clamp(self.log_s, min=-10, max=10)

        return (
            torch.stack(Z_prime_list),    # (B, 768)
            torch.stack(Z_proj_list),     # (B, 768)
            torch.stack(z_partial_list),  # (B, 768)
            s_chunks_list,                # list[Tensor(N, 768)]
            t_chunks_list,                # list[Tensor(N, 768)]
            torch.stack(mu_list),         # (B, 256)
            torch.stack(sigma_list),      # (B, 256)
            log_s,                        # (4,)
        )


class DocumentClassifier(nn.Module):
    """Used at fine-tuning. Loads encoder + teacher attention from GlocalIBModel checkpoint."""

    def __init__(
        self,
        encoder,
        tokenizer,
        attn_pool,                    # pass model.attn_pool_teacher
        num_labels: int = 10,
        max_paragraphs: int = 50,
        device: str = "cuda",
    ):
        super().__init__()
        self.encoder       = encoder
        self.tokenizer     = tokenizer
        self.attn_pool     = attn_pool
        self.classifier    = nn.Linear(768, num_labels)
        self.max_paragraphs = max_paragraphs
        self.to(device)

    def _encode_paragraphs(self, paragraphs: list[str]) -> torch.Tensor:
        """Identical logic to GlocalIBModel._encode_paragraphs (stop_grad=False)."""
        CHUNK_SIZE = 510
        sub_chunks, boundaries = [], []
        for para in paragraphs:
            ids   = self.tokenizer.encode(para, add_special_tokens=False)
            start = len(sub_chunks)
            if len(ids) <= CHUNK_SIZE:
                sub_chunks.append(para)
            else:
                for i in range(0, len(ids), CHUNK_SIZE):
                    sub_chunks.append(self.tokenizer.decode(ids[i:i + CHUNK_SIZE]))
            boundaries.append((start, len(sub_chunks)))

        enc = self.tokenizer(
            sub_chunks, padding=True, truncation=True,
            max_length=512, return_tensors="pt",
        ).to(self.encoder.device)
        cls_vecs = self.encoder(**enc).last_hidden_state[:, 0, :]
        return torch.stack([cls_vecs[s:e].mean(0) for s, e in boundaries])

    def forward(self, paragraphs_batch: list[list[str]]) -> torch.Tensor:
        doc_vecs = torch.stack([
            self.attn_pool(self._encode_paragraphs(paras))
            for paras in paragraphs_batch
        ])                                              # (B, 768)
        return torch.sigmoid(self.classifier(doc_vecs)) # (B, num_labels)
```

---

### `src/loss.py` — Updated

```python
import torch
import torch.nn.functional as F


def alignment_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """1 − mean cosine similarity. z1, z2: (B, D) or flattened."""
    return 1.0 - F.cosine_similarity(z1, z2, dim=-1).mean()


def compression_loss(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """KL( N(mu,sigma²) ∥ N(0,1) ) closed form. mu, sigma: (B, H)."""
    return -0.5 * (1 + 2 * torch.log(sigma) - mu.pow(2) - sigma.pow(2)).sum(-1).mean()


def glocal_ib_loss(
    Z_prime:   torch.Tensor,        # (B, 768) teacher full-doc
    Z_proj:    torch.Tensor,        # (B, 768) student post-IB
    z_partial: torch.Tensor,        # (B, 768) student pre-IB (for L_inter)
    s_chunks:  list,                # list[Tensor(N, 768)] student all-para reps
    t_chunks:  list,                # list[Tensor(N, 768)] teacher all-para reps
    mu:        torch.Tensor,        # (B, 256)
    sigma:     torch.Tensor,        # (B, 256)
    log_s:     torch.Tensor,        # (4,) clamped UW weights
):
    """
    Four-component hierarchical GlocalIB loss.
    Returns (total, l_compress, l_local, l_inter, l_global).
    """
    l_compress = compression_loss(mu, sigma)
    l_local    = alignment_loss(torch.cat(s_chunks), torch.cat(t_chunks))
    l_inter    = alignment_loss(z_partial, Z_prime.detach())
    l_global   = alignment_loss(Z_proj, Z_prime.detach())

    losses = torch.stack([l_compress, l_local, l_inter, l_global])
    total  = (losses * torch.exp(-log_s) + log_s).sum()

    return total, l_compress, l_local, l_inter, l_global
```

---

### `scripts/02_pretrain_glocal.py` — Updated Training Loop

```python
import torch
import wandb
from torch.optim import AdamW
from torch.utils.data import DataLoader
from accelerate import Accelerator
import sys

sys.path.append(".")
from src.data import load_ecthr, mask_text, get_paragraph_mask
from src.model import GlocalIBModel
from src.loss import glocal_ib_loss

# ── CONFIG ────────────────────────────────────────────────────────────────────
EPOCHS        = 5
BATCH_SIZE    = 1          # per GPU; effective = BATCH_SIZE × num_GPUs × GRAD_ACCUM
GRAD_ACCUM    = 4
LR            = 1e-5
MAX_GRAD_NORM = 1.0
EMA_TAU       = 0.99
WANDB_PROJECT = "glocal-nlp"
CONDITION     = "glocal_ib"
# ─────────────────────────────────────────────────────────────────────────────


def collate_fn(batch):
    return [item["text"] for item in batch]


def train():
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=GRAD_ACCUM,
    )
    device = accelerator.device

    if accelerator.is_main_process:
        wandb.init(project=WANDB_PROJECT, name=CONDITION, config={
            "condition": CONDITION, "epochs": EPOCHS,
            "batch_size": BATCH_SIZE, "grad_accum": GRAD_ACCUM,
            "lr": LR, "ema_tau": EMA_TAU,
        })

    dataset = load_ecthr()
    model   = GlocalIBModel(ema_tau=EMA_TAU, device=str(device))
    opt     = AdamW(model.parameters(), lr=LR)

    loader = DataLoader(
        dataset["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    model, opt, loader = accelerator.prepare(model, opt, loader)

    global_step = 0
    for epoch in range(EPOCHS):
        model.train()
        for full_batch in loader:
            # Build Xm: word/sentence mask each paragraph for each document
            masked_batch = [
                [mask_text(para) for para in doc]
                for doc in full_batch
            ]
            # Paragraph dropout indices (for student pooling stage only)
            kept_indices_batch = [
                get_paragraph_mask(len(doc))
                for doc in full_batch
            ]

            with accelerator.accumulate(model):
                out = model(full_batch, masked_batch, kept_indices_batch)
                total, lc, ll, li, lg = glocal_ib_loss(*out)
                accelerator.backward(total)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

                opt.step()
                opt.zero_grad()

                # EMA update after optimizer step
                if accelerator.sync_gradients:
                    unwrapped = accelerator.unwrap_model(model)
                    unwrapped.update_teacher_ema()

            if accelerator.sync_gradients:
                global_step += 1
                if global_step % 10 == 0 and accelerator.is_main_process:
                    log_s = out[7].detach().float()
                    wandb.log({
                        "total_loss":       total.item(),
                        "l_compress":       lc.item(),
                        "l_local":          ll.item(),
                        "l_inter":          li.item(),
                        "l_global":         lg.item(),
                        "log_s_compress":   log_s[0].item(),
                        "log_s_local":      log_s[1].item(),
                        "log_s_inter":      log_s[2].item(),
                        "log_s_global":     log_s[3].item(),
                        "epoch":            epoch,
                        "step":             global_step,
                    })

        if accelerator.is_main_process:
            print(f"Epoch {epoch} done.")
            accelerator.save_state(f"checkpoints/{CONDITION}_epoch{epoch}")
            torch.save(
                accelerator.unwrap_model(model).state_dict(),
                f"checkpoints/{CONDITION}_epoch{epoch}.pt",
            )

    if accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    train()
```

---

### `scripts/03_pretrain_mlm.py` — MLM Baseline (unchanged from v2)

Uses `distilroberta-base` with standard HuggingFace `DataCollatorForLanguageModeling`
(token-level masking, 15%, no span or sentence masking). This is the comparison baseline —
keeping MLM masking vanilla ensures any performance difference is attributable to the
pre-training objective, not the masking strategy.

---

## SLURM Scripts

### `slurm/pretrain_glocal.sh`

```bash
#!/bin/bash
#SBATCH --job-name=glocal_pretrain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --partition=spark
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/glocal_%j.out
#SBATCH --error=logs/glocal_%j.err

source ~/.bashrc
conda activate glocal_nlp
cd /path/to/GlocalDoc

accelerate launch --num_processes=4 scripts/02_pretrain_glocal.py
```

### `slurm/pretrain_mlm.sh`

```bash
#!/bin/bash
#SBATCH --job-name=mlm_pretrain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --partition=spark
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/mlm_%j.out
#SBATCH --error=logs/mlm_%j.err

source ~/.bashrc
conda activate glocal_nlp
cd /path/to/GlocalDoc

accelerate launch --num_processes=4 scripts/03_pretrain_mlm.py
```

### `slurm/finetune.sh`

```bash
#!/bin/bash
#SBATCH --job-name=finetune
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/finetune_%j.out
#SBATCH --error=logs/finetune_%j.err

source ~/.bashrc
conda activate glocal_nlp
cd /path/to/GlocalDoc

python scripts/04_finetune.py
```

---

## Memory Budget

Target: stable training at BATCH_SIZE=1 (effective 4) on a 32GB GPU.

| Optimization | Savings |
|---|---|
| distilroberta-base (82M vs 125M) | ~34% parameter memory |
| Gradient checkpointing (6 layers) | ~50% activation memory per encoder pass |
| Sequential teacher/student encoding | Peak VRAM = one pass (N paras), not two simultaneously |
| Dynamic padding (`padding=True`) | ~40% average token reduction |
| bf16 via accelerate | ~40% activation memory |
| BATCH_SIZE=1 + GRAD_ACCUM=4 | Peak capped at one doc per step |
| Paragraph cap (max 50) | Bounded attention pooling input |

On Spark (4× A100 80GB): increase `BATCH_SIZE=4`, remove `GRAD_ACCUM`.

---

## Key Invariants

- Encoder loads as `RobertaModel.from_pretrained("distilroberta-base")` — no `DistilRobertaModel`.
- Teacher encoder is always stop-grad: `torch.no_grad()` inside `_encode_paragraphs(stop_grad=True)`.
- Teacher/student encoder passes are sequential — never interleaved — to minimize peak VRAM.
- EMA update happens after every `optimizer.step()`, before `zero_grad()`.
- Paragraph loss (`L_local`) covers **all N paragraph pairs** — paragraph dropout affects pooling only, not encoding.
- `DocumentClassifier` uses `attn_pool_teacher` (EMA-updated, seen full docs throughout training).
- `log_s` is clamped to `[−10, 10]` in every forward pass.
- If all four `log_s` stay near 0 by epoch 2, UW is degenerate — investigate before proceeding.
- MLM baseline uses vanilla token masking (15%) — no span or sentence masking.

| Condition | Pre-training | Masking |
|-----------|-------------|---------|
| `glocal_ib` | Full 4-component loss + UW + EMA attention | span + sentence (Xm) + paragraph dropout |
| `mlm` | Standard MLM on ECtHR paragraphs | vanilla 15% token masking |

Fine-tuning: N ∈ {10, 50, 100} × 5 seeds. Metric: **macro-F1**.

---

## Verification Checklist

Run before submitting to cluster:

```bash
# 1. Import check
python -c "
from src.model import GlocalIBModel, DocumentClassifier
from src.loss import glocal_ib_loss
from src.data import load_ecthr, mask_text, get_paragraph_mask
print('imports OK')
"

# 2. Masking smoke test
python -c "
from src.data import mask_text, get_paragraph_mask
para = 'The applicant was arrested on 3 March 1998. He was held without charge for six days. The court found this violated Article 5.'
masked = mask_text(para)
print('original:', para)
print('masked:  ', masked)
indices = get_paragraph_mask(20)
print('kept indices (30% dropped from 20):', indices, '— count:', len(indices))
"

# 3. Forward pass shape check (CPU OK)
python -c "
import torch
from src.model import GlocalIBModel
from src.data import mask_text, get_paragraph_mask

model = GlocalIBModel(device='cpu')
full  = [['Para A. Sentence one. Sentence two.', 'Para B. More content here.', 'Para C.', 'Para D.', 'Para E.']]
masked = [[mask_text(p) for p in doc] for doc in full]
kept   = [get_paragraph_mask(len(doc)) for doc in full]
Z_prime, Z_proj, z_partial, s_chunks, t_chunks, mu, sigma, log_s = model(full, masked, kept)
assert Z_prime.shape  == (1, 768)
assert Z_proj.shape   == (1, 768)
assert z_partial.shape == (1, 768)
assert mu.shape       == (1, 256)
assert log_s.shape    == (4,)
print('shapes OK')
"

# 4. Full forward+backward pass
python -c "
import torch
from src.model import GlocalIBModel
from src.loss import glocal_ib_loss
from src.data import mask_text, get_paragraph_mask
from torch.optim import AdamW

model = GlocalIBModel(device='cpu')
opt   = AdamW(model.parameters(), lr=1e-5)
full  = [['Para A. Sentence one. Sentence two.', 'Para B.', 'Para C.', 'Para D.', 'Para E.']]
masked = [[mask_text(p) for p in doc] for doc in full]
kept   = [get_paragraph_mask(len(doc)) for doc in full]

opt.zero_grad()
out = model(full, masked, kept)
total, lc, ll, li, lg = glocal_ib_loss(*out)
assert torch.isfinite(total), f'loss NaN/Inf: {total}'
total.backward()
opt.step()
model.update_teacher_ema()
print(f'OK — total={total.item():.4f} compress={lc.item():.4f} local={ll.item():.4f} inter={li.item():.4f} global={lg.item():.4f}')
"

# 5. Smoke test: 5 training steps (should not OOM on 32GB)
# python scripts/02_pretrain_glocal.py   # Ctrl-C after ~5 steps
```

---

## Literature Cited

| Paper | Used for |
|-------|---------|
| Yang et al. (2025) — *GlocalIB* | Original framework |
| Tishby et al. (2000) — *IB principle* | Theoretical foundation |
| Alemi et al. (2017) — *Deep VIB* | Variational IB, reparameterization |
| Grill et al. (2020) — *BYOL* | EMA teacher / stop-gradient motivation |
| Kendall, Gal & Cipolla (CVPR 2018) — *Homoscedastic UW* | Loss weight parameterization |
| Kirchdorfer et al. (IJCV 2025) — *UW-SO* | Escalation path if UW degenerates |
| Chalkidis et al. (ACL 2022) — *LexGLUE* | ECtHR dataset and benchmark |
| Yang et al. (NAACL 2016) — *HAN* | Attention pooling over paragraphs |
