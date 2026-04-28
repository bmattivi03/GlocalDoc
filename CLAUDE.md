# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MSc research project at Free University of Bozen-Bolzano (April 2026). Adapts the GlocalIB (Global-Local Information Bottleneck) objective from time series imputation to few-shot legal document classification using the ECtHR dataset.

The core bet: forcing a model to reconstruct a full document's meaning from partially masked paragraphs — with an explicit compression bottleneck — produces representations that generalize better with only 10–100 labeled examples.

## Environment Setup

```bash
conda create -n glocal_nlp python=3.10
conda activate glocal_nlp
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install transformers datasets accelerate wandb
```

GPU is required for pre-training. Verify with:
```python
python -c "import torch; print(torch.cuda.is_available())"
```

## Architecture

Two-branch setup sharing RoBERTa-base weights:

**Teacher branch** (frozen, stop-gradient): encodes the full document. Produces deterministic Z' (768-dim CLS mean-pool over all paragraphs).

**Student branch** (trainable): encodes the same document with 20–40% of paragraphs randomly dropped. Outputs probabilistic (mu, sigma) via a head on the pooled representation. Samples Z via reparameterization, then projects to Z_proj via MLP.

```
Probabilistic head:   CLS (768) → Linear → mu (256)
                      CLS (768) → Linear → log_sigma → sigma = exp(log_sigma)
                      Z = mu + sigma * N(0,1)

MLP projector:        Z (256) → Linear(256,512) → ReLU → Linear(512,768) → Z_proj

Learnable beta:       log_beta = nn.Parameter(torch.tensor(0.0))
                      beta = clamp(exp(log_beta), min=0.01)
```

**Chunk-and-pool**: RoBERTa has a 512-token limit, so each paragraph is encoded independently (CLS token → 768-dim), then paragraph vectors are mean-pooled into one document vector.

## Loss

```
L = L_alignment + beta * L_compression
L_alignment   = 1 - cosine_similarity(Z_proj, Z')
L_compression = KL( N(mu, sigma^2) || N(0,1) )
              = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
```

KL is closed-form — no approximation needed.

## Dataset

ECtHR via HuggingFace LexGLUE:

```python
from datasets import load_dataset
dataset = load_dataset("coastalcph/lex_glue", "ecthr_a")
# Each example: {"text": [paragraph1, paragraph2, ...], "labels": [article_ids]}
```

Data cleaning (only filter applied — no hard paragraph cap in final version):
```python
dataset = dataset.filter(lambda x: len(x["text"]) >= 5)
```

The 100-paragraph cap in `import.ipynb` was exploratory — the spec says very long documents should be kept. The teacher reads all paragraphs; the student reads a masked subset.

## Experimental Conditions

Three conditions, identical everything except the training objective:

| Condition | Description |
|---|---|
| MLM baseline | RoBERTa + standard masked language modeling on ECtHR |
| GlocalIB beta=0 | Full GlocalIB architecture, IB term disabled |
| GlocalIB full | Full architecture with learnable beta (our method) |

Fine-tuning: N = {10, 50, 100} labeled examples × 5 seeds. Metric: macro-F1 on ECtHR test set.

## Key Invariants

- Teacher branch must always see the full document (all paragraphs). Stop-gradient prevents collapse.
- Student paragraph masking (20–40% dropped) is independent of the 512-token chunk limit.
- Beta lower bound of 0.01 prevents compression collapse — do not remove it.
- Evaluation metric is macro-F1 (class imbalance: Article 3 has 4704 cases, Article 5 has 41).
- Report beta trajectory during training as a diagnostic — if it collapses to the floor (0.01) consistently, the IB term is not doing real work.

## Logging

Use Weights & Biases (`wandb`) for experiment tracking. Log: training loss, L_alignment, L_compression, beta value, and macro-F1 per seed.
