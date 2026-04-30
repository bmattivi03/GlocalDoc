# GlocalDoc v2 — Implementation Plan

> **Self-contained.** This document can be followed from a fresh clone with no prior context.
> Supersedes all earlier plans. Last updated 2026-04-29.

---

## Background

**Project:** MSc research at Free University of Bozen-Bolzano.
Adapts the GlocalIB (Global-Local Information Bottleneck) objective from time series imputation to
few-shot legal document classification on the ECtHR dataset (LexGLUE / `coastalcph/lex_glue`).

**Why this redesign:**
The first training attempt (2026-04-29, RTX 5000, 32GB) OOM'd immediately.
Root cause: each paragraph was encoded in a sequential Python `for` loop,
preventing GPU batching and keeping all activations alive at once.
With RoBERTa-base (125M params, 12 layers) and ECtHR docs averaging ~40 paragraphs, peak VRAM
explodes before a single optimizer step completes.

**The fix is not one tweak — it is four changes applied together:**

| Axis | Before | After |
|------|--------|-------|
| Encoder | RoBERTa-base (125M) | DistilRoBERTa-base (82M) |
| Encoding | Sequential per-paragraph loop | Single batched forward pass |
| Aggregation | Mean pooling | Attention pooling + positional embeddings |
| Loss weights | `clamp(exp(log_w), 0.01)` — collapse-prone | Homoscedastic Uncertainty Weighting |

---

## Repository Layout

```
GlocalDoc/
├── src/
│   ├── __init__.py
│   ├── data.py          ← unchanged (mask_paragraphs, load_ecthr, sample_few_shot)
│   ├── model.py         ← FULL REWRITE
│   └── loss.py          ← FULL REWRITE
├── train_glocal.py      ← UPDATE
├── train_mlm.py         ← UPDATE (swap to DistilRoBERTa)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_pretrain_glocal.ipynb
│   ├── 03_pretrain_mlm.ipynb
│   ├── 04_finetune.ipynb
│   └── 05_evaluate.ipynb
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
python -c "from transformers import DistilRobertaModel; print('transformers OK')"
```

**`requirements.txt`** must contain (update if missing):
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

## Single Machine vs GPU Cluster — No Code Change Needed

**Short answer:** The code structure is identical for both environments. The `accelerate` library
(already a dependency) handles the difference through configuration, not code.

**How it works:**

```python
# train_glocal.py — the same file runs in all cases
accelerator = Accelerator(
    mixed_precision="bf16",           # halves activation memory; ignored on CPU
    gradient_accumulation_steps=4,    # effective batch = BATCH_SIZE * num_GPUs * 4
)
```

| Scenario | How to run |
|----------|-----------|
| Single machine, 1 GPU | `python train_glocal.py` |
| Single machine, debug on CPU | `python train_glocal.py` (bf16 gracefully ignored) |
| Cluster, 4 GPUs (DDP) | `accelerate launch --num_processes=4 train_glocal.py` |
| Cluster via SLURM + torchrun | `torchrun --nproc_per_node=4 train_glocal.py` |

`Accelerator` auto-detects the device count and DDP environment. The model, optimizer, and
dataloader are wrapped with `accelerator.prepare(...)` once — DDP, gradient sync, and device
placement are handled transparently.

**One-time accelerate config (optional, recommended for multi-GPU):**
```bash
accelerate config   # interactive wizard; saves ~/.cache/huggingface/accelerate/default_config.yaml
```

---

## Architecture Specification

### 1. Encoder: `distilroberta-base`

```
HuggingFace ID:  "distilroberta-base"
Parameters:       ~82M  (vs 125M RoBERTa — 34% smaller, ~50% faster)
Layers:           6     (vs 12)
Hidden dim:       768   (unchanged — all downstream heads stay identical)
Tokenizer:        RobertaTokenizerFast  (same vocabulary as RoBERTa)
```

Gradient checkpointing enabled: `encoder.gradient_checkpointing_enable()`

### 2. Batched Paragraph Encoding

```python
def _encode_chunks(self, paragraphs: list[str], stop_grad: bool, max_chunks: int = 50):
    # Truncate long documents (keep first + last halves)
    if len(paragraphs) > max_chunks:
        half = max_chunks // 2
        paragraphs = paragraphs[:half] + paragraphs[-half:]

    # Batch tokenize — dynamic padding to actual max length in batch (NOT max_length=512)
    enc = self.tokenizer(
        paragraphs,
        padding=True,          # pads to actual max in this batch
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(self.encoder.device)

    # Single batched forward — one pass for all N paragraphs
    ctx = torch.no_grad() if stop_grad else contextlib.nullcontext()
    with ctx:
        out = self.encoder(**enc)   # (N, seq_len, 768)

    return out.last_hidden_state[:, 0, :]   # (N, 768) — CLS token per chunk
```

Why this matters: `padding=True` (dynamic) vs `padding="max_length"` reduces average sequence
length by ~40%. Combined with batch processing (vs loop), this is the primary memory fix.

### 3. Aggregation: `AttentionPooling`

```python
class AttentionPooling(nn.Module):
    def __init__(self, dim: int = 768, max_chunks: int = 200):
        super().__init__()
        self.attn_query = nn.Parameter(torch.randn(dim) * 0.01)
        self.chunk_pos  = nn.Embedding(max_chunks, dim)
        nn.init.zeros_(self.chunk_pos.weight)   # zero-init: starts as mean pooling

    def forward(self, chunk_vecs: torch.Tensor, return_weights: bool = False):
        # chunk_vecs: (N, 768)
        N = chunk_vecs.size(0)
        pos_ids = torch.arange(N, device=chunk_vecs.device)
        vecs    = chunk_vecs + self.chunk_pos(pos_ids)   # add positional information

        scores  = vecs @ self.attn_query                  # (N,)
        weights = torch.softmax(scores, dim=0)            # (N,)
        doc_vec = (weights.unsqueeze(-1) * vecs).sum(0)   # (768,)

        if return_weights:
            return doc_vec, weights
        return doc_vec
```

**Literature support:** Query-based attention pooling over chunk CLS vectors outperforms mean
pooling by 1–3% on document classification tasks (multiple 2024–2025 papers). Positional
embeddings improve performance when document structure is semantic — which it is for ECtHR
(strict factual/legal/reasoning section order). Zero-init ensures the model starts equivalent
to mean pooling and learns deviations.

Teacher and student **share the same** `AttentionPooling` instance. This is intentional: the
pooling learns document-general paragraph importance that benefits both branches.

### 4. Four-Component Loss

```
L_total = L_compress × exp(−s₀) + s₀
        + L_local   × exp(−s₁) + s₁
        + L_inter   × exp(−s₂) + s₂
        + L_global  × exp(−s₃) + s₃
```

`log_s = [s₀, s₁, s₂, s₃]` are four learnable parameters, initialized at 0 (σᵢ = 1 initially).

| Component | Formula | Purpose |
|-----------|---------|---------|
| **L_compress** | KL(N(μ,σ²) ∥ N(0,1)) | IB penalty — regularizes latent space |
| **L_local** | cosine_dist(s_chunks_kept, t_chunks_kept) | Per-chunk fidelity |
| **L_inter** | cosine_dist(attn_pool(s_chunks), attn_pool(t_chunks_kept)) | Partial-doc aggregate without bottleneck |
| **L_global** | cosine_dist(Z_proj, Z_prime) | Full-doc reconstruction through IB bottleneck |

Variables:
- `s_chunks` = student chunk CLS vecs for kept/visible paragraphs
- `t_chunks_kept` = teacher chunk CLS vecs indexed to the same kept positions
- `Z_prime` = teacher full-doc vector (attention pool over **all** chunks, stop-gradient)
- `Z_proj` = student output through probabilistic head + reparameterization + MLP projector

**Ablation mode** (`disable_ib=True`): return only `L_global`, skip IB and weighted terms.
Used for the `glocal_beta0` experimental condition.

### 5. Homoscedastic Uncertainty Weighting

**Why:** Naive `clamp(exp(log_w), min=0.01)` is collapse-prone. The model can drive `log_w`
large to inflate weights, then adjust gradient norms to compensate. The clamp does not prevent
this asymptotic failure.

**How Kendall & Gal (CVPR 2018) prevents collapse:**
```
L_total = Σᵢ [ Lᵢ × exp(−sᵢ) + sᵢ ]
```
Increasing `sᵢ` (→ larger σ) reduces `exp(−sᵢ)` (task weight goes down) but increases `sᵢ`
(regularization cost goes up) at the same rate. These forces balance — there is no free lunch.
The gradient of `sᵢ` w.r.t. total loss is zero only at the true optimum.

**Implementation:**
```python
# In model: four scalars, init at 0
self.log_s = nn.Parameter(torch.zeros(4))

# In loss:
losses = torch.stack([l_compress, l_local, l_inter, l_global])
total  = (losses * torch.exp(-self.log_s) + self.log_s).sum()
```

**Diagnostic:** Log all four `log_s` values at every W&B step. If they all stay near 0 by
epoch 2, the weighting is degenerate — escalate to UW-SO (Kirchdorfer et al., IJCV 2025).

### 6. Probabilistic Head (unchanged from v1)

```
CLS (768) → Linear(768, 256) → μ
CLS (768) → Linear(768, 256) → log_σ → σ = exp(log_σ)
Z = μ + σ × ε,   ε ~ N(0, I)     [reparameterization]
Z (256) → Linear(256, 512) → ReLU → Linear(512, 768) → Z_proj
```

---

## File Implementations

### `src/model.py` — Full Rewrite

```python
import contextlib
import torch
import torch.nn as nn
from transformers import DistilRobertaModel, RobertaTokenizerFast


class AttentionPooling(nn.Module):
    def __init__(self, dim: int = 768, max_chunks: int = 200):
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
    def __init__(self, hidden_dim: int = 256, proj_dim: int = 512,
                 max_chunks: int = 50, device: str = "cuda"):
        super().__init__()
        self.tokenizer = RobertaTokenizerFast.from_pretrained("distilroberta-base")
        self.encoder   = DistilRobertaModel.from_pretrained("distilroberta-base")
        self.encoder.gradient_checkpointing_enable()

        self.attention_pool = AttentionPooling(dim=768, max_chunks=200)

        # IB probabilistic head
        self.mu_head        = nn.Linear(768, hidden_dim)
        self.log_sigma_head = nn.Linear(768, hidden_dim)

        # MLP projector: 256 → 512 → 768
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, 768),
        )

        # Homoscedastic uncertainty weights: [compress, local, inter, global]
        self.log_s = nn.Parameter(torch.zeros(4))

        self.max_chunks = max_chunks
        self.to(device)

    def _encode_chunks(self, paragraphs: list[str], stop_grad: bool) -> torch.Tensor:
        if len(paragraphs) > self.max_chunks:
            half = self.max_chunks // 2
            paragraphs = paragraphs[:half] + paragraphs[-half:]

        enc = self.tokenizer(
            paragraphs, padding=True, truncation=True,
            max_length=512, return_tensors="pt",
        ).to(self.encoder.device)

        ctx = torch.no_grad() if stop_grad else contextlib.nullcontext()
        with ctx:
            out = self.encoder(**enc)

        return out.last_hidden_state[:, 0, :]   # (N, 768)

    def forward(self, full_batch: list[list[str]],
                masked_batch: list[list[str]],
                kept_indices_batch: list[list[int]]):

        Z_prime_list, Z_proj_list  = [], []
        Z_inter_s_list, Z_inter_t_list = [], []
        chunks_s_list, chunks_t_list   = [], []
        mu_list, sigma_list = [], []

        for full, masked, indices in zip(full_batch, masked_batch, kept_indices_batch):
            # Teacher branch (stop-gradient)
            t_chunks = self._encode_chunks(full, stop_grad=True)          # (N, 768)
            Z_prime  = self.attention_pool(t_chunks)                      # (768,)
            valid_idx   = [i for i in indices if i < len(t_chunks)]
            if not valid_idx:
                valid_idx = [0]
            t_chunks_kept = t_chunks[valid_idx]                           # (M, 768)

            # Student branch
            s_chunks  = self._encode_chunks(masked, stop_grad=False)      # (M, 768)
            z_partial = self.attention_pool(s_chunks)                     # (768,)

            # IB bottleneck
            mu       = self.mu_head(z_partial)
            sigma    = torch.exp(self.log_sigma_head(z_partial))
            z_sample = mu + sigma * torch.randn_like(mu)
            Z_proj   = self.projector(z_sample)                           # (768,)

            # Teacher partial pool (for intermediate loss)
            Z_teacher_partial = self.attention_pool(t_chunks_kept)        # (768,)

            Z_prime_list.append(Z_prime)
            Z_proj_list.append(Z_proj)
            Z_inter_s_list.append(z_partial)
            Z_inter_t_list.append(Z_teacher_partial)
            chunks_s_list.append(s_chunks)
            chunks_t_list.append(t_chunks_kept)
            mu_list.append(mu)
            sigma_list.append(sigma)

        return (
            torch.stack(Z_prime_list),       # (B, 768) — teacher full doc
            torch.stack(Z_proj_list),        # (B, 768) — student IB projection
            torch.stack(Z_inter_s_list),     # (B, 768) — student partial aggregate
            torch.stack(Z_inter_t_list),     # (B, 768) — teacher partial aggregate
            chunks_s_list,                   # list[Tensor(M, 768)] — variable length
            chunks_t_list,                   # list[Tensor(M, 768)] — variable length
            torch.stack(mu_list),            # (B, 256)
            torch.stack(sigma_list),         # (B, 256)
            self.log_s,                      # (4,) — learnable weights
        )


class DocumentClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, tokenizer,
                 num_labels: int = 10, max_chunks: int = 50, device: str = "cuda"):
        super().__init__()
        self.encoder     = encoder
        self.tokenizer   = tokenizer
        self.attn_pool   = AttentionPooling(dim=768, max_chunks=200)
        self.classifier  = nn.Linear(768, num_labels)
        self.max_chunks  = max_chunks
        self.to(device)

    def _encode_chunks(self, paragraphs: list[str]) -> torch.Tensor:
        if len(paragraphs) > self.max_chunks:
            half = self.max_chunks // 2
            paragraphs = paragraphs[:half] + paragraphs[-half:]
        enc = self.tokenizer(
            paragraphs, padding=True, truncation=True,
            max_length=512, return_tensors="pt",
        ).to(self.encoder.device)
        out = self.encoder(**enc)
        return out.last_hidden_state[:, 0, :]   # (N, 768)

    def forward(self, paragraphs_batch: list[list[str]]) -> torch.Tensor:
        doc_vecs = torch.stack([
            self.attn_pool(self._encode_chunks(paras))
            for paras in paragraphs_batch
        ])   # (B, 768)
        return torch.sigmoid(self.classifier(doc_vecs))   # (B, num_labels)
```

### `src/loss.py` — Full Rewrite

```python
import torch
import torch.nn.functional as F


def alignment_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """1 − mean cosine similarity. Inputs: (N, D)."""
    if z1.dim() == 3:
        z1 = z1.view(-1, z1.size(-1))
        z2 = z2.view(-1, z2.size(-1))
    return 1.0 - F.cosine_similarity(z1, z2, dim=-1).mean()


def compression_loss(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """KL( N(mu, sigma²) ∥ N(0,1) ) — closed form. Inputs: (B, H)."""
    return -0.5 * torch.sum(
        1 + 2 * torch.log(sigma) - mu.pow(2) - sigma.pow(2), dim=-1
    ).mean()


def glocal_ib_loss(
    Z_prime:     torch.Tensor,           # (B, 768) — teacher full doc
    Z_proj:      torch.Tensor,           # (B, 768) — student IB projection
    Z_inter_s:   torch.Tensor,           # (B, 768) — student partial aggregate
    Z_inter_t:   torch.Tensor,           # (B, 768) — teacher partial aggregate
    chunks_s:    list,                   # list[Tensor(M, 768)]
    chunks_t:    list,                   # list[Tensor(M, 768)]
    mu:          torch.Tensor,           # (B, 256)
    sigma:       torch.Tensor,           # (B, 256)
    log_s:       torch.Tensor,           # (4,)  learnable
    disable_ib:  bool = False,
):
    """
    Four-component hierarchical GlocalIB loss with homoscedastic uncertainty weighting.

    Returns: (total, l_compress, l_local, l_inter, l_global)
    """
    l_compress = compression_loss(mu, sigma)
    l_local    = alignment_loss(torch.cat(chunks_s), torch.cat(chunks_t))
    l_inter    = alignment_loss(Z_inter_s, Z_inter_t)
    l_global   = alignment_loss(Z_proj, Z_prime)

    if disable_ib:
        return l_global, torch.zeros(1), torch.zeros(1), torch.zeros(1), l_global

    # Homoscedastic uncertainty weighting (Kendall, Gal & Cipolla, CVPR 2018)
    losses = torch.stack([l_compress, l_local, l_inter, l_global])
    total  = (losses * torch.exp(-log_s) + log_s).sum()

    return total, l_compress, l_local, l_inter, l_global
```

### `train_glocal.py` — Updated

```python
import torch
import wandb
from torch.optim import AdamW
from torch.utils.data import DataLoader
from accelerate import Accelerator
import sys

sys.path.append(".")
from src.data import load_ecthr, mask_paragraphs
from src.model import GlocalIBModel
from src.loss import glocal_ib_loss

# --- CONFIG ---
EPOCHS         = 5
BATCH_SIZE     = 1      # per GPU; effective = BATCH_SIZE × num_GPUs × GRAD_ACCUM_STEPS
GRAD_ACCUM     = 4
LR             = 1e-5
MAX_GRAD_NORM  = 1.0
WANDB_PROJECT  = "glocal-nlp"
CONDITION      = "glocal_ib"   # "glocal_ib" or "glocal_beta0"
DISABLE_IB     = (CONDITION == "glocal_beta0")


def collate_fn(batch):
    return [item["text"] for item in batch]


def train():
    accelerator = Accelerator(
        mixed_precision="bf16",             # halves activation memory; transparent on CPU
        gradient_accumulation_steps=GRAD_ACCUM,
    )
    device = accelerator.device

    if accelerator.is_main_process:
        wandb.init(project=WANDB_PROJECT, name=CONDITION, config={
            "condition": CONDITION, "epochs": EPOCHS,
            "batch_size": BATCH_SIZE, "grad_accum": GRAD_ACCUM, "lr": LR,
        })

    dataset = load_ecthr()
    model   = GlocalIBModel(device=str(device))
    opt     = AdamW(model.parameters(), lr=LR)

    loader = DataLoader(
        dataset["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    model, opt, loader = accelerator.prepare(model, opt, loader)

    global_step = 0
    for epoch in range(EPOCHS):
        model.train()
        for i, full_batch in enumerate(loader):
            masked_data    = [mask_paragraphs(text) for text in full_batch]
            masked_batch   = [m[0] for m in masked_data]
            indices_batch  = [m[1] for m in masked_data]

            with accelerator.accumulate(model):
                out = model(full_batch, masked_batch, indices_batch)
                total, lc, ll, li, lg = glocal_ib_loss(*out, disable_ib=DISABLE_IB)
                accelerator.backward(total)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                opt.step()
                opt.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                if global_step % 10 == 0 and accelerator.is_main_process:
                    log_s = out[8].detach().float()
                    wandb.log({
                        "total_loss": total.item(),
                        "l_compress": lc.item(),
                        "l_local":    ll.item(),
                        "l_inter":    li.item(),
                        "l_global":   lg.item(),
                        "log_s_0_compress": log_s[0].item(),
                        "log_s_1_local":    log_s[1].item(),
                        "log_s_2_inter":    log_s[2].item(),
                        "log_s_3_global":   log_s[3].item(),
                        "epoch": epoch, "step": global_step,
                    })

        if accelerator.is_main_process:
            print(f"Epoch {epoch} done.")
            accelerator.save_state(f"checkpoints/{CONDITION}_epoch{epoch}")

    if accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    train()
```

**To run:**
```bash
# Single GPU
python train_glocal.py

# 4 GPUs
accelerate launch --num_processes=4 train_glocal.py

# Via SLURM (see slurm/ scripts)
```

### `train_mlm.py` — Update Encoder Only

Replace:
```python
from transformers import RobertaForMaskedLM, RobertaTokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model     = RobertaForMaskedLM.from_pretrained("roberta-base")
```
With:
```python
from transformers import RobertaForMaskedLM, RobertaTokenizerFast
tokenizer = RobertaTokenizerFast.from_pretrained("distilroberta-base")
model     = RobertaForMaskedLM.from_pretrained("distilroberta-base")
```
Everything else stays the same (DistilRoBERTa shares the RoBERTa tokenizer vocabulary and
uses `RobertaForMaskedLM` — it is architecturally a 6-layer RoBERTa).

---

## SLURM Scripts

### `slurm/pretrain_glocal.sh`

```bash
#!/bin/bash
#SBATCH --job-name=glocal_pretrain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --partition=spark           # ← confirm partition name with lab admin
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/glocal_%j.out
#SBATCH --error=logs/glocal_%j.err

source ~/.bashrc
conda activate glocal_nlp
cd /path/to/GlocalDoc               # ← update before submitting

# Edit CONDITION in train_glocal.py before each run
accelerate launch --num_processes=4 train_glocal.py
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

accelerate launch --num_processes=4 train_mlm.py
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

jupyter nbconvert --to notebook --execute notebooks/04_finetune.ipynb \
    --output notebooks/04_finetune_executed.ipynb \
    --ExecutePreprocessor.timeout=86400
```

---

## Memory Budget Estimate

With all optimizations on a 32GB GPU:

| Optimization | Approximate savings |
|---|---|
| RoBERTa → DistilRoBERTa | ~34% parameter memory |
| Gradient checkpointing (6 layers) | ~50% activation memory per encoder pass |
| Batched encoding | Eliminates sequential CUDA overhead |
| Dynamic padding (`padding=True`) | ~40% average token reduction |
| bf16 via accelerate | ~40% activation memory |
| BATCH_SIZE=1 + GRAD_ACCUM=4 | Peak memory capped at 1 doc per step |

**Target:** Stable training at batch=1 (effective 4) on 32GB.
On Spark (4× A100 80GB): increase `BATCH_SIZE=4` and remove `GRAD_ACCUM`.

---

## Key Invariants

- Teacher always sees the **full** document. Stop-gradient enforced via `torch.no_grad()` inside `_encode_chunks(stop_grad=True)`.
- Evaluation metric: **macro-F1** (ECtHR has severe class imbalance: Article 3 has 4704 training cases, Article 5 has 41).
- If all four `log_s` values converge to ~0 by epoch 2, report as diagnostic (weighting degenerate) — do not silently ignore.
- Three experimental conditions must share identical data, identical starting weights, only the pre-training objective differs.

| Condition | Pre-training | `disable_ib` |
|-----------|-------------|--------------|
| `glocal_ib` | Full 4-component loss + UW | `False` |
| `glocal_beta0` | Global alignment only | `True` |
| `mlm` | Standard MLM on ECtHR paragraphs | N/A |

---

## Verification Checklist

Run these before submitting to the cluster:

```bash
# 1. Import check
python -c "
from src.model import GlocalIBModel, DocumentClassifier
from src.loss import glocal_ib_loss
from src.data import load_ecthr, mask_paragraphs
print('All imports OK')
"

# 2. Forward pass shape check (CPU is fine for this)
python -c "
import torch
from src.model import GlocalIBModel
from src.data import mask_paragraphs

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model  = GlocalIBModel(device=device)
full   = [['Paragraph one.', 'Paragraph two.', 'Paragraph three.', 'Paragraph four.']]
md     = [mask_paragraphs(p) for p in full]
out    = model(full, [m[0] for m in md], [m[1] for m in md])
Z_prime, Z_proj, Zi_s, Zi_t, cs, ct, mu, sigma, log_s = out
assert Z_prime.shape == (1, 768), f'Z_prime shape: {Z_prime.shape}'
assert Z_proj.shape  == (1, 768), f'Z_proj shape:  {Z_proj.shape}'
assert mu.shape      == (1, 256), f'mu shape:      {mu.shape}'
assert log_s.shape   == (4,),     f'log_s shape:   {log_s.shape}'
print('Forward pass shapes OK')
"

# 3. Backward pass — loss must be finite, grad must flow
python -c "
import torch
from src.model import GlocalIBModel
from src.loss import glocal_ib_loss
from src.data import mask_paragraphs
from torch.optim import AdamW

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model  = GlocalIBModel(device=device)
opt    = AdamW(model.parameters(), lr=1e-5)
full   = [['Para A.', 'Para B.', 'Para C.', 'Para D.']]
md     = [mask_paragraphs(p) for p in full]

opt.zero_grad()
out    = model(full, [m[0] for m in md], [m[1] for m in md])
total, lc, ll, li, lg = glocal_ib_loss(*out)
assert torch.isfinite(total), 'Loss is NaN/Inf!'
total.backward()
opt.step()
print(f'Backward OK — total={total.item():.4f}  compress={lc.item():.4f}  local={ll.item():.4f}  inter={li.item():.4f}  global={lg.item():.4f}')
"

# 4. 5-step smoke test (should not OOM on 32GB)
python train_glocal.py   # Ctrl-C after ~5 steps if it's running fine
```

---

## Literature Cited

| Paper | Used for |
|-------|---------|
| Kendall, Gal & Cipolla (CVPR 2018) — *Multi-Task Learning Using Uncertainty to Weigh Losses* | Loss weight parameterization |
| Chen et al. (ICML 2018) — *GradNorm: Gradient Normalization for Adaptive Loss Balancing* | Alternative weight method (rejected: overhead) |
| Kirchdorfer et al. (IJCV 2025) — *Investigating Uncertainty Weighting for Multi-Task Learning* | Escalation path (UW-SO) |
| Yang et al. (NAACL 2016) — *Hierarchical Attention Networks for Document Classification* | Attention pooling design |
| Chalkidis et al. (ACL 2022) — *LexGLUE: A Benchmark Dataset for Legal Language Understanding* | ECtHR hierarchical baselines |
| Samarinas et al. (Nature 2025) — *Explainable judgment prediction via LexFaith BERT* | ECtHR-specific validation of hierarchical approach |
| Jiang et al. (arXiv 2025) — *LMK > CLS: Landmark Pooling for Dense Embeddings* | Alternative aggregation (future work) |
