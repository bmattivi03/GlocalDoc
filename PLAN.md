# GlocalIB NLP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the full GlocalIB NLP pipeline — from data loading through pre-training, fine-tuning, and evaluation — across three experimental conditions (GlocalIB, GlocalIB β=0, MLM baseline) with Jupyter notebooks as the interface and Python modules as the reusable core.

**Architecture:** Two-branch teacher-student setup sharing RoBERTa-base weights. Teacher reads full document (stop-gradient), student reads masked paragraphs and outputs a probabilistic representation (mu, sigma) that is compressed via a learnable beta and projected to align with the teacher. Fine-tuning adds a linear classifier on top of the pre-trained encoder and trains all weights.

**Tech Stack:** Python 3.10, PyTorch, HuggingFace Transformers + Datasets + Trainer, scikit-learn (macro-F1), Weights & Biases, conda (`glocal_nlp`), SLURM (university cluster)

**Compute:**
- Standard runs: NVIDIA GPU 32GB VRAM — use for data exploration, sanity checks, and fine-tuning
- Heavy runs (GlocalIB + MLM pre-training): Spark node 128GB VRAM — submit via SLURM with `--gres=gpu:4` or the cluster-specific Spark partition name (confirm partition name with lab admin)
- Batch size guidance: 32GB GPU → `BATCH_SIZE=4`; Spark 128GB → `BATCH_SIZE=16` (4× scale-up)

---

## File Structure

```
GlocalDoc/
├── src/
│   ├── data.py          # ECtHR loading, filtering, masking, few-shot sampling
│   ├── model.py         # GlocalIBModel + DocumentClassifier
│   └── loss.py          # alignment_loss, compression_loss, glocal_ib_loss
├── notebooks/
│   ├── 01_data_exploration.ipynb   # rename/replace import.ipynb
│   ├── 02_pretrain_glocal.ipynb    # GlocalIB full + beta=0 ablation
│   ├── 03_pretrain_mlm.ipynb       # MLM baseline via HF Trainer
│   ├── 04_finetune.ipynb           # all 3 conditions × N × seeds
│   └── 05_evaluate.ipynb           # macro-F1 table, performance curve, beta plot
├── slurm/
│   ├── pretrain_glocal.sh
│   ├── pretrain_mlm.sh
│   └── finetune.sh
├── checkpoints/                    # saved model weights (gitignored)
├── results/                        # JSON result files
├── logs/                           # SLURM stdout/stderr
├── requirements.txt
└── CLAUDE.md
```

---

## Task 1: Repository Structure

**Files:**
- Create: `src/__init__.py`, `checkpoints/.gitkeep`, `results/.gitkeep`, `logs/.gitkeep`
- Modify: `requirements.txt`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p src notebooks slurm checkpoints results logs
touch src/__init__.py
touch checkpoints/.gitkeep results/.gitkeep logs/.gitkeep
```

- [ ] **Step 2: Update requirements.txt**

```
torch
torchvision
torchaudio
transformers
datasets
scikit-learn
wandb
jupyter
nbconvert
accelerate
```

- [ ] **Step 3: Verify conda env has all packages**

```bash
conda activate glocal_nlp
pip install -r requirements.txt
python -c "import torch, transformers, datasets, sklearn, wandb; print('OK')"
```
Expected output: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/ notebooks/ slurm/ checkpoints/.gitkeep results/.gitkeep logs/.gitkeep requirements.txt
git commit -m "chore: set up project structure"
```

---

## Task 2: Data Pipeline (`src/data.py`)

**Files:**
- Create: `src/data.py`

- [ ] **Step 1: Write `src/data.py`**

```python
import random
import numpy as np
from datasets import load_dataset


def load_ecthr(min_paragraphs=5):
    """Load ECtHR dataset and filter out documents with fewer than min_paragraphs."""
    dataset = load_dataset("coastalcph/lex_glue", "ecthr_a")
    dataset = dataset.filter(lambda x: len(x["text"]) >= min_paragraphs)
    return dataset


def mask_paragraphs(paragraphs, mask_ratio_min=0.2, mask_ratio_max=0.4):
    """Randomly drop 20–40% of paragraphs. Always keeps at least 1."""
    n = len(paragraphs)
    mask_ratio = random.uniform(mask_ratio_min, mask_ratio_max)
    n_keep = max(1, int(n * (1 - mask_ratio)))
    indices = sorted(random.sample(range(n), n_keep))
    return [paragraphs[i] for i in indices]


def sample_few_shot(dataset_split, n_per_class, seed, num_classes=10):
    """
    Sample indices such that each class has at least n_per_class examples.
    Documents can satisfy multiple classes (multi-label). Deduplicates.
    """
    rng = random.Random(seed)
    per_class = [[] for _ in range(num_classes)]
    indices = list(range(len(dataset_split)))
    rng.shuffle(indices)
    selected_set = set()

    for idx in indices:
        labels = dataset_split[idx]["labels"]
        for label in labels:
            if len(per_class[label]) < n_per_class:
                per_class[label].append(idx)
                selected_set.add(idx)
        if all(len(pc) >= n_per_class for pc in per_class):
            break

    return [dataset_split[i] for i in sorted(selected_set)]
```

- [ ] **Step 2: Verify data pipeline in Python**

```python
from src.data import load_ecthr, mask_paragraphs, sample_few_shot

dataset = load_ecthr()
assert len(dataset["train"]) > 8000

ex = dataset["train"][0]
full = ex["text"]
masked = mask_paragraphs(full)
assert 1 <= len(masked) < len(full)
assert 0.6 <= len(masked) / len(full) <= 0.85

few_shot = sample_few_shot(dataset["train"], n_per_class=10, seed=42)
assert len(few_shot) >= 10
print(f"Few-shot size (n=10): {len(few_shot)} examples — OK")
```

Expected: no assertion errors, few-shot size printed.

- [ ] **Step 3: Commit**

```bash
git add src/data.py
git commit -m "feat: add ECtHR data pipeline with masking and few-shot sampling"
```

---

## Task 3: Loss Functions (`src/loss.py`)

**Files:**
- Create: `src/loss.py`

- [ ] **Step 1: Write `src/loss.py`**

```python
import torch
import torch.nn.functional as F


def alignment_loss(z_proj, z_prime):
    """1 - mean cosine similarity. z_proj and z_prime are (B, 768)."""
    return 1.0 - F.cosine_similarity(z_proj, z_prime, dim=-1).mean()


def compression_loss(mu, sigma):
    """
    Closed-form KL( N(mu, sigma^2) || N(0,1) ).
    mu and sigma are (B, hidden_dim).
    """
    return -0.5 * torch.sum(
        1 + 2 * torch.log(sigma) - mu.pow(2) - sigma.pow(2), dim=-1
    ).mean()


def glocal_ib_loss(z_proj, z_prime, mu, sigma, beta, disable_ib=False):
    """
    Total GlocalIB loss.
    disable_ib=True → beta=0 ablation (alignment only, IB term zeroed out).
    Returns: (total_loss, l_align, l_compress)
    """
    l_align = alignment_loss(z_proj, z_prime)
    l_compress = compression_loss(mu, sigma)
    if disable_ib:
        return l_align, l_align, torch.tensor(0.0, device=z_proj.device)
    total = l_align + beta * l_compress
    return total, l_align, l_compress
```

- [ ] **Step 2: Verify losses are finite on random inputs**

```python
import torch
from src.loss import alignment_loss, compression_loss, glocal_ib_loss

B, D, H = 4, 768, 256
z_proj  = torch.randn(B, D)
z_prime = torch.randn(B, D)
mu      = torch.randn(B, H)
sigma   = torch.abs(torch.randn(B, H)) + 1e-6
beta    = torch.tensor(0.1)

l_a = alignment_loss(z_proj, z_prime)
l_c = compression_loss(mu, sigma)
total, la, lc = glocal_ib_loss(z_proj, z_prime, mu, sigma, beta)

assert torch.isfinite(l_a)
assert torch.isfinite(l_c)
assert torch.isfinite(total)
print(f"l_align={la:.4f}, l_compress={lc:.4f}, total={total:.4f} — OK")
```

- [ ] **Step 3: Verify beta=0 ablation returns l_align only**

```python
total_abl, la_abl, lc_abl = glocal_ib_loss(z_proj, z_prime, mu, sigma, beta, disable_ib=True)
assert torch.allclose(total_abl, la_abl)
assert lc_abl.item() == 0.0
print("beta=0 ablation — OK")
```

- [ ] **Step 4: Commit**

```bash
git add src/loss.py
git commit -m "feat: add GlocalIB loss (alignment + KL compression + beta=0 ablation flag)"
```

---

## Task 4: Model Architecture (`src/model.py`)

**Files:**
- Create: `src/model.py`

- [ ] **Step 1: Write `src/model.py`**

```python
import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer


class GlocalIBModel(nn.Module):
    """
    Teacher-student GlocalIB model.
    - Teacher: full document, stop-gradient (no_grad on encoder forward)
    - Student: masked document, probabilistic head, MLP projector, learnable beta
    Both branches share self.encoder weights. Teacher path does not contribute gradients.
    """

    def __init__(self, hidden_dim=256, proj_dim=512, device="cuda"):
        super().__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.encoder = RobertaModel.from_pretrained("roberta-base")

        # Student probabilistic head
        self.mu_head = nn.Linear(768, hidden_dim)
        self.log_sigma_head = nn.Linear(768, hidden_dim)

        # MLP projector: Z(256) → 512 → 768
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, 768),
        )

        # Learnable compression strength; clamped to min 0.01
        self.log_beta = nn.Parameter(torch.tensor(0.0))

        self.hidden_dim = hidden_dim
        self.device = device
        self.to(device)

    def _encode_paragraphs(self, paragraphs, stop_grad=False):
        """
        Encode a list of paragraph strings to a (768,) document vector.
        Each paragraph encoded via RoBERTa CLS token, then mean-pooled (chunk-and-pool).
        stop_grad=True: teacher branch — gradients do not flow through encoder.
        """
        para_vecs = []
        for para in paragraphs:
            enc = self.tokenizer(
                para,
                max_length=512,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            ).to(self.device)
            if stop_grad:
                with torch.no_grad():
                    out = self.encoder(**enc)
            else:
                out = self.encoder(**enc)
            cls_vec = out.last_hidden_state[:, 0, :]  # (1, 768)
            para_vecs.append(cls_vec)
        return torch.stack(para_vecs).squeeze(1).mean(0)  # (768,)

    def forward(self, full_paragraphs_batch, masked_paragraphs_batch):
        """
        full_paragraphs_batch:   list[list[str]]  — teacher inputs (all paragraphs)
        masked_paragraphs_batch: list[list[str]]  — student inputs (masked paragraphs)
        Returns: Z_prime (B,768), mu (B,256), sigma (B,256), Z_proj (B,768), beta scalar
        """
        batch_z_prime, batch_mu, batch_sigma, batch_z_proj = [], [], [], []

        for full_paras, masked_paras in zip(full_paragraphs_batch, masked_paragraphs_batch):
            # Teacher branch — stop-gradient
            z_prime = self._encode_paragraphs(full_paras, stop_grad=True)   # (768,)

            # Student branch — trainable
            h = self._encode_paragraphs(masked_paras, stop_grad=False)      # (768,)
            mu = self.mu_head(h)                                             # (256,)
            log_sigma = self.log_sigma_head(h)                               # (256,)
            sigma = torch.exp(log_sigma)

            eps = torch.randn_like(mu)
            z = mu + sigma * eps                                             # reparameterization
            z_proj = self.projector(z)                                       # (768,)

            batch_z_prime.append(z_prime)
            batch_mu.append(mu)
            batch_sigma.append(sigma)
            batch_z_proj.append(z_proj)

        Z_prime = torch.stack(batch_z_prime)   # (B, 768)
        mu      = torch.stack(batch_mu)        # (B, 256)
        sigma   = torch.stack(batch_sigma)     # (B, 256)
        Z_proj  = torch.stack(batch_z_proj)    # (B, 768)
        beta    = torch.clamp(torch.exp(self.log_beta), min=0.01)

        return Z_prime, mu, sigma, Z_proj, beta


class DocumentClassifier(nn.Module):
    """
    Fine-tuning classifier. Full fine-tuning: all encoder weights + linear head trainable.
    Uses same chunk-and-pool as GlocalIBModel for consistency.
    """

    def __init__(self, encoder, tokenizer, num_labels=10, device="cuda"):
        super().__init__()
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.classifier = nn.Linear(768, num_labels)
        self.device = device
        self.to(device)

    def _encode_paragraphs(self, paragraphs):
        para_vecs = []
        for para in paragraphs:
            enc = self.tokenizer(
                para,
                max_length=512,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            ).to(self.device)
            out = self.encoder(**enc)
            cls_vec = out.last_hidden_state[:, 0, :]
            para_vecs.append(cls_vec)
        return torch.stack(para_vecs).squeeze(1).mean(0)  # (768,)

    def forward(self, paragraphs_batch):
        """paragraphs_batch: list[list[str]] → (B, num_labels) sigmoid probabilities"""
        doc_vecs = torch.stack([
            self._encode_paragraphs(paras) for paras in paragraphs_batch
        ])  # (B, 768)
        return torch.sigmoid(self.classifier(doc_vecs))  # (B, 10)
```

- [ ] **Step 2: Verify GlocalIBModel forward pass is finite**

```python
import torch
from src.model import GlocalIBModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model  = GlocalIBModel(device=DEVICE)

full_batch   = [["The court observed the applicant.", "Article 3 was invoked."]]
masked_batch = [["The court observed the applicant."]]

Z_prime, mu, sigma, Z_proj, beta = model(full_batch, masked_batch)

assert Z_prime.shape == (1, 768)
assert mu.shape      == (1, 256)
assert sigma.shape   == (1, 256)
assert Z_proj.shape  == (1, 768)
assert torch.isfinite(Z_prime).all()
assert torch.isfinite(Z_proj).all()
assert beta.item() >= 0.01
print(f"Forward pass OK — beta={beta.item():.4f}")
```

- [ ] **Step 3: Verify backward pass**

```python
from src.loss import glocal_ib_loss
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=1e-5)
optimizer.zero_grad()
Z_prime, mu, sigma, Z_proj, beta = model(full_batch, masked_batch)
loss, _, _ = glocal_ib_loss(Z_proj, Z_prime, mu, sigma, beta)
loss.backward()
optimizer.step()
assert torch.isfinite(loss)
print(f"Backward pass OK — loss={loss.item():.4f}")
```

- [ ] **Step 4: Verify DocumentClassifier**

```python
from src.model import DocumentClassifier
from transformers import RobertaModel, RobertaTokenizer

encoder   = RobertaModel.from_pretrained("roberta-base").to(DEVICE)
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
clf       = DocumentClassifier(encoder, tokenizer, device=DEVICE)

logits = clf([["The court found a violation.", "Article 6 was discussed."]])
assert logits.shape == (1, 10)
assert (logits >= 0).all() and (logits <= 1).all()
print(f"DocumentClassifier OK")
```

- [ ] **Step 5: Commit**

```bash
git add src/model.py
git commit -m "feat: add GlocalIBModel (teacher-student + probabilistic head) and DocumentClassifier"
```

---

## Task 5: Data Exploration Notebook

**Files:**
- Create: `notebooks/01_data_exploration.ipynb` (replaces `import.ipynb`)

- [ ] **Step 1: Create `notebooks/01_data_exploration.ipynb` with these cells**

Cell 1:
```python
import sys
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from src.data import load_ecthr, mask_paragraphs
```

Cell 2:
```python
dataset = load_ecthr()
print(f"Train: {len(dataset['train'])} | Val: {len(dataset['validation'])} | Test: {len(dataset['test'])}")
print("Example keys:", dataset["train"][0].keys())
print("Labels example:", dataset["train"][0]["labels"])
print("N paragraphs example:", len(dataset["train"][0]["text"]))
```

Cell 3:
```python
lengths = [len(ex["text"]) for ex in dataset["train"]]
print(f"Avg paragraphs: {np.mean(lengths):.1f}")
print(f"Max paragraphs: {np.max(lengths)}")
print(f"Min paragraphs: {np.min(lengths)}")

plt.hist(lengths, bins=50)
plt.xlabel("Paragraphs per document")
plt.ylabel("Count")
plt.title("ECtHR paragraph length distribution (train)")
plt.tight_layout()
plt.show()
```

Cell 4:
```python
all_labels = [l for ex in dataset["train"] for l in ex["labels"]]
counter = Counter(all_labels)
print("Label distribution:", dict(sorted(counter.items())))
# Article 3 (index 3) dominates → macro-F1 required
```

Cell 5:
```python
ex     = dataset["train"][0]
full   = ex["text"]
masked = mask_paragraphs(full)
print(f"Full: {len(full)} paragraphs | Masked: {len(masked)} paragraphs")
print(f"Drop ratio: {1 - len(masked)/len(full):.2%}")
```

- [ ] **Step 2: Run notebook top-to-bottom**

```bash
jupyter nbconvert --to notebook --execute notebooks/01_data_exploration.ipynb \
    --output notebooks/01_data_exploration.ipynb --ExecutePreprocessor.timeout=300
```

- [ ] **Step 3: Commit**

```bash
git add notebooks/01_data_exploration.ipynb
git rm import.ipynb
git commit -m "feat: data exploration notebook with paragraph stats and masking verification"
```

---

## Task 6: GlocalIB Pre-training Notebook

**Files:**
- Create: `notebooks/02_pretrain_glocal.ipynb`

- [ ] **Step 1: Create `notebooks/02_pretrain_glocal.ipynb` with these cells**

Cell 1 — Config (edit `CONDITION` before each run):
```python
import sys, os, torch
sys.path.append("..")

CONDITION      = "glocal_ib"   # "glocal_ib" or "glocal_beta0"
DISABLE_IB     = (CONDITION == "glocal_beta0")
EPOCHS         = 5
BATCH_SIZE     = 4
LR             = 1e-5
CHECKPOINT_DIR = "../checkpoints"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
print(f"Condition: {CONDITION} | Device: {DEVICE} | disable_ib: {DISABLE_IB}")
```

Cell 2 — Setup:
```python
import random
import wandb
from torch.optim import AdamW
from src.data import load_ecthr, mask_paragraphs
from src.model import GlocalIBModel
from src.loss import glocal_ib_loss

wandb.init(project="glocal-nlp", name=CONDITION, config={
    "condition": CONDITION, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
})

dataset    = load_ecthr()
train_data = dataset["train"]
model      = GlocalIBModel(device=DEVICE)
optimizer  = AdamW(model.parameters(), lr=LR)
```

Cell 3 — Training loop:
```python
for epoch in range(EPOCHS):
    model.train()
    indices    = list(range(len(train_data)))
    random.shuffle(indices)
    epoch_loss = 0.0
    n_steps    = 0

    for i in range(0, len(indices), BATCH_SIZE):
        batch_idx    = indices[i : i + BATCH_SIZE]
        full_batch   = [train_data[j]["text"] for j in batch_idx]
        masked_batch = [mask_paragraphs(train_data[j]["text"]) for j in batch_idx]

        optimizer.zero_grad()
        Z_prime, mu, sigma, Z_proj, beta = model(full_batch, masked_batch)
        loss, l_align, l_compress = glocal_ib_loss(
            Z_proj, Z_prime, mu, sigma, beta, disable_ib=DISABLE_IB
        )
        loss.backward()
        optimizer.step()

        wandb.log({
            "loss": loss.item(), "l_align": l_align.item(),
            "l_compress": l_compress.item(), "beta": beta.item(), "epoch": epoch,
        })
        epoch_loss += loss.item()
        n_steps    += 1

    avg = epoch_loss / n_steps
    print(f"Epoch {epoch+1}/{EPOCHS} — avg_loss: {avg:.4f} | beta: {beta.item():.4f}")

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "beta": beta.item(),
    }, f"{CHECKPOINT_DIR}/{CONDITION}_epoch{epoch+1}.pt")

wandb.finish()
print("Pre-training complete.")
```

- [ ] **Step 2: Local sanity run (2 examples, 1 epoch)**

Add before the training loop (remove after verifying):
```python
train_data = [train_data[0], train_data[1]]
```

Run: `jupyter nbconvert --to notebook --execute notebooks/02_pretrain_glocal.ipynb --ExecutePreprocessor.timeout=300`

Expected: loss printed, checkpoint saved at `checkpoints/glocal_ib_epoch1.pt`, no NaN.

- [ ] **Step 3: Commit**

```bash
git add notebooks/02_pretrain_glocal.ipynb
git commit -m "feat: GlocalIB pre-training notebook (full + beta=0 ablation via CONDITION flag)"
```

---

## Task 7: MLM Baseline Notebook

**Files:**
- Create: `notebooks/03_pretrain_mlm.ipynb`

- [ ] **Step 1: Create `notebooks/03_pretrain_mlm.ipynb` with these cells**

Cell 1:
```python
import sys
sys.path.append("..")
from datasets import load_dataset
from transformers import (
    RobertaTokenizerFast, RobertaForMaskedLM,
    DataCollatorForLanguageModeling, Trainer, TrainingArguments,
)

CHECKPOINT_DIR = "../checkpoints/mlm"
EPOCHS         = 5
BATCH_SIZE     = 8
```

Cell 2 — Load and flatten:
```python
raw = load_dataset("coastalcph/lex_glue", "ecthr_a")
raw = raw.filter(lambda x: len(x["text"]) >= 5)

def flatten(example):
    return {"text": " ".join(example["text"])}

flat = raw.map(flatten, remove_columns=["labels"])
print(f"Train size: {len(flat['train'])}")
```

Cell 3 — Tokenize:
```python
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

def tokenize_fn(examples):
    return tokenizer(
        examples["text"], truncation=True, max_length=512, padding="max_length"
    )

tokenized = flat.map(tokenize_fn, batched=True, remove_columns=["text"])
tokenized.set_format("torch")
```

Cell 4 — Train:
```python
model         = RobertaForMaskedLM.from_pretrained("roberta-base")
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

training_args = TrainingArguments(
    output_dir                  = CHECKPOINT_DIR,
    num_train_epochs            = EPOCHS,
    per_device_train_batch_size = BATCH_SIZE,
    save_steps                  = 1000,
    logging_steps               = 100,
    report_to                   = "wandb",
    run_name                    = "mlm_baseline",
)

trainer = Trainer(
    model         = model,
    args          = training_args,
    train_dataset = tokenized["train"],
    data_collator = data_collator,
)

trainer.train()
trainer.save_model(f"{CHECKPOINT_DIR}/mlm_final")
print("MLM pre-training complete.")
```

- [ ] **Step 2: Local sanity run (add `max_steps=2` to TrainingArguments, remove after)**

- [ ] **Step 3: Commit**

```bash
git add notebooks/03_pretrain_mlm.ipynb
git commit -m "feat: MLM baseline pre-training notebook via HuggingFace Trainer"
```

---

## Task 8: SLURM Job Scripts

**Files:**
- Create: `slurm/pretrain_glocal.sh`, `slurm/pretrain_mlm.sh`, `slurm/finetune.sh`

- [ ] **Step 1: Write `slurm/pretrain_glocal.sh`** (targets Spark 128GB node for pre-training)

```bash
#!/bin/bash
#SBATCH --job-name=glocal_pretrain
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=spark         # ← confirm partition name with lab admin
#SBATCH --gres=gpu:4              # 4× GPU on Spark = 128GB total VRAM
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/glocal_%j.out
#SBATCH --error=logs/glocal_%j.err

source ~/.bashrc
conda activate glocal_nlp
cd /path/to/GlocalDoc             # ← update before submitting

# Edit CONDITION in Cell 1 before each run (glocal_ib or glocal_beta0)
# On Spark: set BATCH_SIZE=16 in the notebook config cell
jupyter nbconvert --to notebook --execute notebooks/02_pretrain_glocal.ipynb \
    --output notebooks/02_pretrain_glocal_executed.ipynb \
    --ExecutePreprocessor.timeout=86400 \
    --ExecutePreprocessor.kernel_name=python3
```

- [ ] **Step 2: Write `slurm/pretrain_mlm.sh`** (targets Spark 128GB node)

```bash
#!/bin/bash
#SBATCH --job-name=mlm_pretrain
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=spark         # ← confirm partition name with lab admin
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/mlm_%j.out
#SBATCH --error=logs/mlm_%j.err

source ~/.bashrc
conda activate glocal_nlp
cd /path/to/GlocalDoc             # ← update before submitting

# On Spark: increase per_device_train_batch_size to 32 in the notebook config cell
jupyter nbconvert --to notebook --execute notebooks/03_pretrain_mlm.ipynb \
    --output notebooks/03_pretrain_mlm_executed.ipynb \
    --ExecutePreprocessor.timeout=86400 \
    --ExecutePreprocessor.kernel_name=python3
```

- [ ] **Step 3: Write `slurm/finetune.sh`** (standard 32GB GPU — fine-tuning is lighter)

```bash
#!/bin/bash
#SBATCH --job-name=finetune
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1              # single 32GB GPU is sufficient for fine-tuning
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/finetune_%j.out
#SBATCH --error=logs/finetune_%j.err

source ~/.bashrc
conda activate glocal_nlp
cd /path/to/GlocalDoc             # ← update before submitting

jupyter nbconvert --to notebook --execute notebooks/04_finetune.ipynb \
    --output notebooks/04_finetune_executed.ipynb \
    --ExecutePreprocessor.timeout=86400 \
    --ExecutePreprocessor.kernel_name=python3
```

- [ ] **Step 4: Commit**

```bash
git add slurm/
git commit -m "feat: SLURM job scripts for pre-training and fine-tuning"
```

---

## Task 9: Fine-tuning Notebook

**Files:**
- Create: `notebooks/04_finetune.ipynb`

- [ ] **Step 1: Create `notebooks/04_finetune.ipynb` with these cells**

Cell 1 — Config:
```python
import sys, os, json, torch
sys.path.append("..")

N_LIST        = [10, 50, 100]
SEEDS         = [0, 1, 2, 3, 4]
EPOCHS        = 10
LR            = 2e-5
RESULTS_DIR   = "../results"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"Device: {DEVICE}")
```

Cell 2 — Imports and helpers:
```python
import numpy as np
from torch.optim import AdamW
from sklearn.metrics import f1_score
from transformers import RobertaTokenizer, RobertaForMaskedLM
from src.data import load_ecthr, sample_few_shot
from src.model import GlocalIBModel, DocumentClassifier


def load_glocal_encoder(checkpoint_path, device):
    glocal = GlocalIBModel(device=device)
    ckpt   = torch.load(checkpoint_path, map_location=device)
    glocal.load_state_dict(ckpt["model_state_dict"])
    return glocal.encoder, glocal.tokenizer


def load_mlm_encoder(checkpoint_dir, device):
    mlm       = RobertaForMaskedLM.from_pretrained(checkpoint_dir)
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    return mlm.roberta.to(device), tokenizer


def train_and_eval(encoder, tokenizer, train_split, test_split, n_per_class, seed, device):
    few_shot  = sample_few_shot(train_split, n_per_class, seed)
    clf       = DocumentClassifier(encoder, tokenizer, device=device)
    optimizer = AdamW(clf.parameters(), lr=LR)
    criterion = torch.nn.BCELoss()

    clf.train()
    for _ in range(EPOCHS):
        for example in few_shot:
            optimizer.zero_grad()
            logits = clf([example["text"]])
            labels = torch.zeros(1, 10).to(device)
            for l in example["labels"]:
                labels[0][l] = 1.0
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

    clf.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for example in test_split:
            pred = clf([example["text"]]).cpu().numpy()[0]
            all_preds.append((pred >= 0.5).astype(int))
            lv = np.zeros(10, dtype=int)
            for l in example["labels"]:
                lv[l] = 1
            all_labels.append(lv)

    return f1_score(np.array(all_labels), np.array(all_preds), average="macro")
```

Cell 3 — Checkpoint paths (edit after pre-training completes):
```python
CONDITIONS = {
    "glocal_ib":    ("../checkpoints/glocal_ib_epoch5.pt",    "glocal"),
    "glocal_beta0": ("../checkpoints/glocal_beta0_epoch5.pt", "glocal"),
    "mlm":          ("../checkpoints/mlm/mlm_final",          "mlm"),
}
```

Cell 4 — Run all experiments:
```python
dataset = load_ecthr()
results = {}

for condition, (ckpt_path, ckpt_type) in CONDITIONS.items():
    if ckpt_type == "glocal":
        encoder, tokenizer = load_glocal_encoder(ckpt_path, DEVICE)
    else:
        encoder, tokenizer = load_mlm_encoder(ckpt_path, DEVICE)

    results[condition] = {}
    for n in N_LIST:
        scores = []
        for seed in SEEDS:
            f1 = train_and_eval(
                encoder, tokenizer,
                dataset["train"], dataset["test"],
                n, seed, DEVICE
            )
            scores.append(f1)
            print(f"{condition} | n={n} | seed={seed} | F1={f1:.4f}")
        results[condition][str(n)] = scores

with open(f"{RESULTS_DIR}/finetuning_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Fine-tuning complete. Results saved.")
```

- [ ] **Step 2: Sanity-run one condition, n=10, seed=0**

Temporarily replace Cell 4 with:
```python
dataset    = load_ecthr()
encoder, tokenizer = load_glocal_encoder(CONDITIONS["glocal_ib"][0], DEVICE)
f1 = train_and_eval(encoder, tokenizer, dataset["train"], dataset["test"], 10, 0, DEVICE)
print(f"Sanity F1: {f1:.4f}")  # expected: a finite float in [0, 1]
```

- [ ] **Step 3: Commit**

```bash
git add notebooks/04_finetune.ipynb
git commit -m "feat: fine-tuning notebook — all conditions × N × seeds, saves results JSON"
```

---

## Task 10: Evaluation Notebook

**Files:**
- Create: `notebooks/05_evaluate.ipynb`

- [ ] **Step 1: Create `notebooks/05_evaluate.ipynb` with these cells**

Cell 1:
```python
import json
import numpy as np
import matplotlib.pyplot as plt

with open("../results/finetuning_results.json") as f:
    results = json.load(f)

CONDITIONS = ["glocal_ib", "glocal_beta0", "mlm"]
N_LIST     = [10, 50, 100]
LABELS     = {
    "glocal_ib":    "GlocalIB full (ours)",
    "glocal_beta0": "GlocalIB β=0 (ablation)",
    "mlm":          "MLM baseline",
}
```

Cell 2 — Results table:
```python
print(f"{'Condition':<25} {'N=10':>14} {'N=50':>14} {'N=100':>14}")
print("-" * 70)
for cond in CONDITIONS:
    row = f"{LABELS[cond]:<25}"
    for n in N_LIST:
        scores = results[cond][str(n)]
        row   += f"  {np.mean(scores):.3f}±{np.std(scores):.3f}"
    print(row)
```

Cell 3 — Performance vs N curve:
```python
fig, ax = plt.subplots(figsize=(8, 5))
markers = {"glocal_ib": "o", "glocal_beta0": "s", "mlm": "^"}
for cond in CONDITIONS:
    means = [np.mean(results[cond][str(n)]) for n in N_LIST]
    stds  = [np.std(results[cond][str(n)])  for n in N_LIST]
    ax.errorbar(N_LIST, means, yerr=stds, label=LABELS[cond],
                marker=markers[cond], capsize=4, linewidth=1.5)

ax.set_xlabel("Labeled examples per class (N)")
ax.set_ylabel("Macro-F1")
ax.set_xscale("log")
ax.set_xticks(N_LIST)
ax.set_xticklabels([str(n) for n in N_LIST])
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("../results/performance_curve.png", dpi=150)
plt.show()
```

Cell 4 — Beta trajectory from W&B:
```python
import wandb

api     = wandb.Api()
run     = api.run("YOUR_ENTITY/glocal-nlp/glocal_ib")  # ← update entity
history = run.history(keys=["beta", "_step"])

plt.figure(figsize=(8, 4))
plt.plot(history["_step"], history["beta"])
plt.xlabel("Training step")
plt.ylabel("β (learned compression strength)")
plt.title("Beta trajectory during GlocalIB pre-training")
plt.axhline(y=0.01, color="r", linestyle="--", label="lower bound (0.01)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("../results/beta_trajectory.png", dpi=150)
plt.show()
```

Cell 5 — Ablation: is the IB term doing real work?
```python
for n in N_LIST:
    glocal_mean = np.mean(results["glocal_ib"][str(n)])
    beta0_mean  = np.mean(results["glocal_beta0"][str(n)])
    delta       = glocal_mean - beta0_mean
    print(f"N={n:>3}: GlocalIB={glocal_mean:.3f}, β=0={beta0_mean:.3f}, Δ={delta:+.3f}")
# Positive Δ at N=10 confirms the IB compression term is doing real work
```

- [ ] **Step 2: Commit**

```bash
git add notebooks/05_evaluate.ipynb
git commit -m "feat: evaluation notebook — macro-F1 table, performance curve, beta trajectory"
```

---

## Verification Checklist

End-to-end sanity before submitting SLURM jobs:

- [ ] `python -c "from src.data import load_ecthr, mask_paragraphs, sample_few_shot; print('data OK')"` passes
- [ ] `python -c "from src.model import GlocalIBModel, DocumentClassifier; print('model OK')"` passes
- [ ] `python -c "from src.loss import glocal_ib_loss; print('loss OK')"` passes
- [ ] Loss is finite and non-NaN after step 0 of GlocalIB training
- [ ] Beta starts near `exp(0.0) = 1.0` and remains ≥ 0.01 throughout training
- [ ] MLM Trainer completes at least 2 steps without error
- [ ] `finetuning_results.json` contains all three conditions with N=10/50/100 and 5 seeds each
- [ ] Performance curve PNG generated without errors
- [ ] GlocalIB full macro-F1 at N=10 > MLM baseline macro-F1 at N=10 (primary hypothesis)
- [ ] GlocalIB full macro-F1 at N=10 > GlocalIB β=0 macro-F1 at N=10 (ablation confirms IB term)
