import sys
import os
import numpy as np
from collections import Counter

sys.path.append(".")
from src.data import load_ecthr, get_paragraph_mask, mask_text

print("Loading dataset...")
dataset = load_ecthr()
print(f"Train: {len(dataset['train'])} | Val: {len(dataset['validation'])} | Test: {len(dataset['test'])}")

ex = dataset["train"][0]
print(f"\nExample — labels: {ex['labels']}  paragraphs: {len(ex['text'])}")

# Paragraph length stats
lengths = [len(ex["text"]) for ex in dataset["train"]]
print(f"\nParagraph count (train):")
print(f"  Avg : {np.mean(lengths):.1f}")
print(f"  Max : {np.max(lengths)}")
print(f"  Min : {np.min(lengths)}")

# Label distribution
all_labels = [l for ex in dataset["train"] for l in ex["labels"]]
dist = dict(sorted(Counter(all_labels).items()))
print(f"\nLabel distribution: {dist}")
print("(Article 3 = label 3 has ~4704 cases, Article 5 = label 5 has ~41 — severe imbalance)")

# Masking sanity check — uses v3 API: word/sentence masking + paragraph dropout
full = ex["text"]
kept_indices = get_paragraph_mask(len(full))
masked_paras = [mask_text(p) for p in full]
print(f"\nMasking check on first training doc:")
print(f"  Full         : {len(full)} paragraphs")
print(f"  Kept (pool)  : {len(kept_indices)} ({len(kept_indices)/len(full):.0%})")
print(f"  Indices      : {kept_indices[:8]}{'...' if len(kept_indices) > 8 else ''}")
print(f"  Para 0 word-masked preview: {masked_paras[0][:120]}...")
assert len(kept_indices) >= 1, "Paragraph dropout removed all paragraphs!"
assert len(kept_indices) < len(full), "Paragraph dropout kept everything — check dropout_rate!"
print("Sanity checks passed!")

# Save histogram (no display — safe for SSH)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.hist(lengths, bins=50, color="skyblue", edgecolor="black")
    plt.xlabel("Paragraphs per document")
    plt.ylabel("Count")
    plt.title("ECtHR Paragraph Length Distribution (Train)")
    plt.grid(axis="y", alpha=0.3)
    plt.savefig("results/paragraph_distribution.png", dpi=150, bbox_inches="tight")
    print("\nHistogram saved → results/paragraph_distribution.png")
except ImportError:
    print("\nmatplotlib not available, skipping histogram")
