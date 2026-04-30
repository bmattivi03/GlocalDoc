# GlocalIB for Low-Resource Legal NLP
**IB-Regularized Pre-training with Learnable Compression for Few-Shot Document Classification**

> MSc Computing for Data Science — Free University of Bozen-Bolzano — April 2026

---

## Core Idea

Take RoBERTa-base. Add a GlocalIB training phase on unlabeled legal documents — forcing the model to reconstruct the global meaning of a document from incomplete paragraphs, with an information bottleneck that compresses away irrelevant noise. Then fine-tune with very few labeled examples.

The teacher always reads the full document. The student reads the same document with some paragraphs removed. The student must reconstruct what the full document means. To do this well, it cannot rely on surface patterns — it is forced to learn concepts.

The bet: a model that learned to forget useless things generalizes better when it only has 10–100 labeled examples.


---

## The Problem

Fine-tuning pre-trained language models on few labeled examples is brittle. The model memorizes surface patterns present in the small training set rather than learning the underlying concept.

- Labeling in legal, medical, and administrative domains is expensive — it requires domain experts
- Standard SSL (SimCSE, DeCLUTR) improves representations but does not explicitly discard irrelevant information
- Continued pre-training with MLM reduces domain shift but not noise
- No existing method combines document-level global-local structure with explicit IB compression for NLP

---

## Research Question

> Does a GlocalIB-style information bottleneck objective with a learnable β during domain pre-training improve few-shot document classification compared to standard MLM pre-training, under identical conditions?

---

## Hypothesis

A RoBERTa encoder trained with GlocalIB on unlabeled legal documents — where compression strength β is learned with a lower bound constraint rather than fixed — produces more compressed and stable representations than one trained with MLM on the same data, resulting in higher and more stable classification accuracy at low label counts on ECtHR.

---

## Why GlocalIB

GlocalIB (Yang et al., 2025) was originally proposed for time series imputation. Its core mechanism:

- **Teacher branch**: takes the full original input → same encoder with stop-gradient → deterministic target Z'
- **Student branch**: takes a masked/incomplete input → encoder → probabilistic head → sample Z → MLP projector → Z_proj
- **Loss**: forces Z_proj to align with Z' through an IB objective

The stop-gradient prevents representational collapse (same intuition as BYOL/SimSiam).

The key insight: this global-local structure maps naturally onto documents.

```
Document (global)
├── Paragraph 1 (local)
├── Paragraph 2 (local)   ← student never sees this
├── Paragraph 3 (local)
└── Paragraph 4 (local)   ← student never sees this
```

Teacher sees all 4. Student sees 2. Student must still reconstruct the global meaning.

---

## Information Bottleneck Principle

Introduced by Tishby et al. (2000):

```
min I(X; Z)   subject to   max I(Z; Y)
```

- **Minimize** mutual information between input X and representation Z → throw away noise
- **Maximize** mutual information between representation Z and target Y → keep task-relevant signal

In practice: the model learns to compress. With few labeled examples, less noise means less overfitting.

---

## Architecture

### Full Picture

```
Document X (full)                    Document X° (masked paragraphs)
        |                                        |
        v                                        v
  RoBERTa (frozen)                        RoBERTa (trainable)
  chunk-and-pool                           chunk-and-pool
        |                                        |
        v                                        v
  deterministic Z'                     Probabilistic Head
  [teacher repr]                        outputs (mu, sigma)
        |                                        |
        |                               sample Z ~ N(mu, sigma^2)
        |                                        |
        |                                   MLP Projector
        |                                        |
        +-----------> Alignment Loss <------  Z_proj

                    + beta * KL(N(mu,sigma^2) || N(0,1))

                    beta learned, clamped to min 0.01
```

### Component Breakdown

**RoBERTa-base** — same weights in both branches. Teacher is frozen (stop-grad), student trains.

**Chunk-and-pool** — RoBERTa has a 512-token limit. Each paragraph is encoded independently via the CLS token (768-dim). Paragraph vectors are mean-pooled into a single document vector.

**Teacher branch** — reads the full document. Produces a deterministic representation Z'. No probabilistic head. No gradients flow through it.

**Student branch** — reads the masked document (20–40% of paragraphs randomly dropped). Produces mu and sigma via a probabilistic head on top of RoBERTa. Samples Z ~ N(mu, sigma^2) via reparameterization trick.

**Probabilistic head** — two linear layers on top of the student CLS pool:
```
CLS (768) → Linear → mu (256)
CLS (768) → Linear → log_sigma (256) → sigma = exp(log_sigma)
Z = mu + sigma * epsilon,   epsilon ~ N(0, 1)
```

**MLP projector** — maps sampled Z into the comparison space:
```
Z (256) → Linear(256, 512) → ReLU → Linear(512, 768) → Z_proj
```
Z_proj is compared to Z' (768-dim teacher output).

**Learnable beta** — controls compression strength:
```python
log_beta = nn.Parameter(torch.tensor(0.0))
beta = torch.clamp(torch.exp(log_beta), min=0.01)
```
Lower bound of 0.01 prevents compression collapse.

### Loss Function

```
L = L_alignment + beta * L_compression

L_alignment  = 1 - cosine_similarity(Z_proj, Z')
L_compression = KL( N(mu, sigma^2) || N(0, 1) )
             = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
```

KL divergence is computable in closed form — no approximation needed.

### NLP Adaptation vs Original GlocalIB

| GlocalIB original | This work |
|---|---|
| Masked tokens/nodes | Masked paragraphs (20–40% dropped) |
| Global target = full sequence | Global target = full document (all paragraphs) |
| Encoder = GNN / MLP | Encoder = RoBERTa-base |
| Pooling = graph pooling | Pooling = chunk-then-pool over paragraphs |
| Deterministic encoder | Student: probabilistic encoder (mu, sigma) |
| Fixed beta | Learnable beta with lower bound |

### Masking: Teacher vs Student

This is critical and distinct from the document length cap:

- **Teacher**: always encodes ALL paragraphs of the document → full global representation Z'
- **Student**: encodes the same document with 20–40% of paragraphs randomly dropped → must reconstruct Z' from partial evidence

The filter (removing documents < 5 paragraphs) is a separate engineering decision — not related to masking. Very long documents (500+ paragraphs) are kept as-is: teacher reads all of them, student reads a masked subset.

---

## Experimental Design

### The Only Variable: Training Objective

| | Baseline | Our Method |
|---|---|---|
| Starting weights | RoBERTa-base | RoBERTa-base |
| Unlabeled data | ECtHR (same) | ECtHR (same) |
| Pre-training objective | Standard MLM | GlocalIB + learnable beta |
| Fine-tuning data | N labeled examples | N labeled examples |
| Evaluation | Macro-F1 | Macro-F1 |

Any difference in performance is caused by the training objective — nothing else.

### Ablation: Is the IB Term Doing Real Work?

A reviewer will ask: "Is the improvement from IB compression, or just from the extra architecture?" Answer with:

| Condition | IB term | Architecture |
|---|---|---|
| Baseline (MLM) | No | standard RoBERTa |
| GlocalIB beta=0 | No (disabled) | full GlocalIB arch |
| GlocalIB full | Yes (learned beta) | full GlocalIB arch |

If full GlocalIB beats beta=0 version → the IB compression is doing real work, not just the architecture.

### Label Counts

N = {10, 50, 100} labeled examples per class, repeated over 5 random seeds each.

The performance vs label count curve is the main result. The gap at N=10 is the critical number.

### Evaluation Metric

Macro-F1 on ECtHR test set, averaged over 5 seeds with standard deviation reported.

Low variance = more stable training = better representations.

### Future Extension (if time allows)

Option A: make the teacher branch also probabilistic — both branches output (mu, sigma) and the alignment loss becomes KL(student dist || teacher dist) rather than comparing to a fixed prior. Cleaner theoretically, more complex to train.

---

## Dataset: ECtHR

**European Court of Human Rights** dataset (Chalkidis et al., 2021).

- Each document = list of factual paragraphs — already pre-split, no preprocessing needed
- Labels = ECHR articles allegedly violated (multi-label, 10 classes)
- Labels require legal expertise → low-resource setting is realistic, not artificial
- Publicly available via HuggingFace LexGLUE benchmark

### Loading

```python
from datasets import load_dataset

# Use LexGLUE version — no trust_remote_code required
dataset = load_dataset("coastalcph/lex_glue", "ecthr_a")
# Each example: {"text": [paragraph1, paragraph2, ...], "labels": [article_ids]}
```

### Real Dataset Statistics (verified)

| Metric | Value |
|---|---|
| Avg paragraphs per case | 23.1 |
| Max paragraphs | 558 |
| Min paragraphs | 1 |
| Dominant class | Article 3 (prohibition of torture) — 4,704 cases |
| Rarest class | Article 5 — 41 cases |

### Label Distribution

| Article index | Count | Right |
|---|---|---|
| 3 | 4704 | Prohibition of torture |
| 9 | 1421 | Freedom of thought |
| 2 | 1368 | Right to life |
| 1 | 1349 | Right to fair trial |
| 4 | 710 | Right to family life |
| 0 | 505 | — |
| 6, 8, 7, 5 | <300 | Rare classes |

Heavy imbalance → macro-F1 is mandatory as evaluation metric.

### Data Cleaning

```python
# Remove documents too short to mask meaningfully (< 5 paragraphs)
# This is the only filter applied — no paragraph cap
dataset = dataset.filter(lambda x: len(x["text"]) >= 5)

# Teacher always reads full document
# Student reads same document with 20-40% of paragraphs dropped
```

Very long documents (500+ paragraphs) are kept. Teacher reads all of them. Student reads a masked subset. This is architecturally correct and intentional.

---

## Baselines

| Condition | What it tests |
|---|---|
| RoBERTa + MLM on ECtHR | **Main comparison** — domain adaptation without IB |
| GlocalIB with beta=0 | Ablation — architecture without compression |
| GlocalIB full (learned beta) | Our method |

The critical comparison is **MLM vs GlocalIB full**. Same data, same starting weights, same fine-tuning. Only the objective changes.

---

## Related Work

| Paper | Relevance |
|---|---|
| Tishby et al. (2000) | Original IB principle |
| Alemi et al. (2017) — Deep VIB | Variational IB, reparameterization trick, KL as compression |
| Liu et al. (2022) | CL as IB instantiation — motivates our framing |
| Yang et al. (2025) — GlocalIB | Original paper, time series imputation |
| Gao et al. (2021) — SimCSE | Main SSL NLP baseline |
| Giorgi et al. (2021) — DeCLUTR | Document-level contrastive baseline |
| Chalkidis et al. (2021) — ECtHR | Dataset |
| Chalkidis et al. (2022) — LexGLUE | Legal NLP benchmark standard |
| Deng et al. (2019) | Meta-pretraining for few-shot NLP |

---

## Key Risks

| Risk | Mitigation |
|---|---|
| IB compression does not survive fine-tuning | Measure I(X;Z) before and after fine-tuning |
| beta collapses to lower bound (0.01) | Report beta trajectory during training as diagnostic |
| GlocalIB does not beat MLM baseline | Pivot claim to variance reduction; still publishable |
| Chunk-and-pool loses cross-paragraph dependencies | Ablate with Longformer if compute allows |
| ECtHR label imbalance | Use macro-F1; apply class weighting in fine-tuning |
| Very long documents slow pre-training | Monitor batch time; optionally cap only extreme outliers |

**Anticipated reviewer objection**: "Is the improvement from IB compression or just the extra architecture?"

**Response**: The beta=0 ablation isolates this. If full GlocalIB beats beta=0 version with same architecture, the compression term is doing real work.

---

## Implementation Plan

| Phase | What | Time |
|---|---|---|
| 1 | Setup env, download RoBERTa + ECtHR, verify dataset | 1 week |
| 2 | Implement chunk-and-pool encoder | 1 week |
| 3 | Implement probabilistic head + reparameterization | 1 week |
| 4 | Implement MLP projector + full forward pass | 1 week |
| 5 | Implement GlocalIB loss + learnable beta | 1 week |
| 6 | Implement MLM baseline training loop | 1 week |
| 7 | Full pre-training runs (baseline + GlocalIB + beta=0 ablation) | 2 weeks |
| 8 | Fine-tuning experiments across N = {10, 50, 100} | 2 weeks |
| 9 | beta trajectory analysis + mutual information measurement | 1 week |
| 10 | Writing | 3 weeks |

**Total: ~14 weeks**

---

## Tech Stack

- **Language**: Python 3.10
- **Framework**: PyTorch + HuggingFace Transformers
- **Encoder**: RoBERTa-base (125M parameters)
- **Dataset**: `coastalcph/lex_glue` via HuggingFace Datasets
- **Environment**: Conda (`glocal_nlp`)
- **Compute**: GPU required for pre-training (Colab Pro / university cluster)
- **Logging**: Weights & Biases

---

## Environment Setup

```bash
# Create environment
conda create -n glocal_nlp python=3.10
conda activate glocal_nlp

# PyTorch with GPU
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# NLP libraries
pip install transformers datasets accelerate

# Verify
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Publication Target

**Primary**: EMNLP Workshop on Legal NLP

**Secondary**: ACL Findings, ECML-PKDD

**Immediate**: arXiv preprint after experiments to establish priority

---

## One-Line Pitch

> We adapt GlocalIB's information bottleneck objective to document-level NLP — using a probabilistic student encoder and a frozen deterministic teacher — showing that forcing compression of irrelevant information during pre-training improves few-shot legal document classification, with compression strength learned automatically during training.