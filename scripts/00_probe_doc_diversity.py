"""
Probe ECtHR document-rep diversity in the pre-trained distilroberta-base space.

Hypothesis: legal documents share heavy boilerplate (case headers, "The Court
notes…", Article-X formulations, citation blocks), so their mean-pooled
paragraph CLS vectors are near-constant across documents — making "predict
the mean" a trivial collapsed solution for the BYOL predictor in pre-training.

This script mean-pools paragraph CLS vectors per document (which approximates
the attention pool at init — `attn_query` is initialised to N(0, 0.01²) and
`chunk_pos` to zero, so softmax weights are nearly uniform). It then reports
the inter-document pairwise cosine similarity distribution.

Decision rule:
  median pairwise cos > 0.90  → hypothesis confirmed; teacher centering is the
                                right fix shape (Stage 2 of plan).
  median pairwise cos < 0.70  → hypothesis refuted; collapse is driven by
                                something other than target leakage.
  0.70 ≤ median ≤ 0.90        → ambiguous; centering still worth trying but
                                the fix may need to be paired with predictor
                                changes (Stage 4).
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from transformers import RobertaModel, RobertaTokenizerFast

sys.path.append(".")
from src.data import load_ecthr

N_DOCS       = 200
SEED         = 0
BATCH_PARAS  = 64    # paragraphs per encoder forward
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN      = 512


def mean_pool_doc(model, tokenizer, paragraphs: list) -> torch.Tensor:
    """Encode all paragraphs of a doc, return mean of CLS vectors → (768,)."""
    cls_vecs = []
    for i in range(0, len(paragraphs), BATCH_PARAS):
        batch = paragraphs[i:i + BATCH_PARAS]
        enc = tokenizer(
            batch, padding=True, truncation=True,
            max_length=MAX_LEN, return_tensors="pt",
        ).to(DEVICE)
        with torch.no_grad():
            out = model(**enc).last_hidden_state[:, 0, :]    # (B, 768)
        cls_vecs.append(out)
    return torch.cat(cls_vecs, dim=0).mean(0)                # (768,)


def main():
    print(f"Device: {DEVICE}")
    print("Loading distilroberta-base...")
    tokenizer = RobertaTokenizerFast.from_pretrained("distilroberta-base")
    model     = RobertaModel.from_pretrained("distilroberta-base").to(DEVICE).eval()

    print("Loading ECtHR (train split, filtered)...")
    dataset = load_ecthr()
    train   = dataset["train"]

    rng     = np.random.default_rng(SEED)
    indices = rng.choice(len(train), size=min(N_DOCS, len(train)), replace=False)
    print(f"Encoding {len(indices)} random documents...")

    doc_vecs = []
    for i, idx in enumerate(indices):
        paras = train[int(idx)]["text"]
        doc_vecs.append(mean_pool_doc(model, tokenizer, paras))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(indices)}")

    Z = torch.stack(doc_vecs)                                # (N, 768)
    Z = F.normalize(Z, dim=-1)
    sims = (Z @ Z.T).cpu().numpy()                           # (N, N)

    # Off-diagonal mask
    n = sims.shape[0]
    mask = ~np.eye(n, dtype=bool)
    pairwise = sims[mask]

    print("\n" + "=" * 60)
    print("Inter-document pairwise cosine similarity")
    print("(mean-pooled paragraph CLS vecs, frozen distilroberta-base)")
    print("=" * 60)
    print(f"  N documents : {n}")
    print(f"  N pairs     : {len(pairwise)}")
    print(f"  min         : {pairwise.min():.4f}")
    print(f"  p10         : {np.percentile(pairwise, 10):.4f}")
    print(f"  median      : {np.median(pairwise):.4f}")
    print(f"  mean        : {pairwise.mean():.4f}")
    print(f"  p90         : {np.percentile(pairwise, 90):.4f}")
    print(f"  max         : {pairwise.max():.4f}")
    print(f"  std         : {pairwise.std():.4f}")
    print(f"  frac > 0.90 : {(pairwise > 0.90).mean():.4f}")
    print(f"  frac > 0.95 : {(pairwise > 0.95).mean():.4f}")
    print(f"  frac > 0.99 : {(pairwise > 0.99).mean():.4f}")

    median = float(np.median(pairwise))
    print("\nVerdict:")
    if median > 0.90:
        print(f"  CONFIRMED — median {median:.3f} > 0.90. Z_prime is a near-constant")
        print("  target across ECtHR docs. DINO-style teacher centering (Stage 2)")
        print("  is the right fix shape.")
    elif median < 0.70:
        print(f"  REFUTED — median {median:.3f} < 0.70. Documents are reasonably")
        print("  distinct in pretrained space. The collapse driver is NOT target")
        print("  leakage; revisit hypothesis (predictor over-capacity? IB pressure?).")
    else:
        print(f"  AMBIGUOUS — median {median:.3f} in [0.70, 0.90]. Centering still")
        print("  worth trying, but expect to also need predictor changes (Stage 4).")

    os.makedirs("results", exist_ok=True)
    out_path = "results/doc_diversity_probe.npz"
    np.savez(out_path, pairwise=pairwise, sims=sims, indices=indices)
    print(f"\nSaved raw stats → {out_path}")


if __name__ == "__main__":
    main()
