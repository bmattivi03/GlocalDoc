import sys
import os
import json
import torch
import numpy as np
from torch.optim import AdamW
from sklearn.metrics import f1_score
from transformers import RobertaForMaskedLM, RobertaTokenizerFast

sys.path.append(".")
from src.data import load_ecthr, sample_few_shot
from src.model import GlocalIBModel, DocumentClassifier

# --- CONFIG ---
N_LIST      = [10, 50, 100]
SEEDS       = [0, 1, 2, 3, 4]
FINETUNE_EPOCHS = 10
LR          = 2e-5
RESULTS_DIR = "results"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(RESULTS_DIR, exist_ok=True)

# Checkpoint paths produced by the pre-training scripts.
# glocal conditions: plain .pt file saved at the end of the last epoch.
# mlm condition:     HuggingFace model directory saved by trainer.save_model().
CONDITIONS = {
    "glocal_ib":    ("checkpoints/glocal_ib_epoch4.pt",    "glocal"),
    "glocal_beta0": ("checkpoints/glocal_beta0_epoch4.pt", "glocal"),
    "mlm":          ("checkpoints/mlm_baseline",            "mlm"),
}


def load_encoder(path, ckpt_type):
    if ckpt_type == "glocal":
        # Reconstruct full GlocalIBModel and load the saved state dict,
        # then extract just the encoder and tokenizer for fine-tuning.
        m = GlocalIBModel(device=DEVICE)
        state = torch.load(path, map_location=DEVICE)
        m.load_state_dict(state)
        return m.encoder, m.tokenizer
    else:
        # MLM checkpoint is a full RobertaForMaskedLM saved in HF format.
        mlm = RobertaForMaskedLM.from_pretrained(path)
        tokenizer = RobertaTokenizerFast.from_pretrained("distilroberta-base")
        return mlm.roberta.to(DEVICE), tokenizer


def run_few_shot(encoder, tokenizer, train_split, test_split, n, seed):
    few_shot = sample_few_shot(train_split, n, seed)
    clf = DocumentClassifier(encoder, tokenizer, device=DEVICE)
    opt = AdamW(clf.parameters(), lr=LR)
    criterion = torch.nn.BCELoss()

    clf.train()
    for _ in range(FINETUNE_EPOCHS):
        for ex in few_shot:
            opt.zero_grad()
            probs  = clf([ex["text"]])
            labels = torch.zeros(1, 10, device=DEVICE)
            for l in ex["labels"]:
                if l < 10:
                    labels[0][l] = 1.0
            loss = criterion(probs, labels)
            loss.backward()
            opt.step()

    clf.eval()
    preds, targets = [], []
    with torch.no_grad():
        for ex in test_split:
            p = clf([ex["text"]]).cpu().numpy()[0]
            preds.append((p >= 0.5).astype(int))
            t = np.zeros(10, dtype=int)
            for l in ex["labels"]:
                if l < 10:
                    t[l] = 1
            targets.append(t)

    return f1_score(np.array(targets), np.array(preds), average="macro", zero_division=0)


def main():
    print("Loading dataset...")
    dataset = load_ecthr()
    all_results = {}

    for cond, (path, ckpt_type) in CONDITIONS.items():
        if not os.path.exists(path):
            print(f"Skipping {cond} — checkpoint not found at {path}")
            continue

        print(f"\n=== Condition: {cond} ===")
        encoder, tokenizer = load_encoder(path, ckpt_type)
        all_results[cond] = {}

        for n in N_LIST:
            scores = []
            for seed in SEEDS:
                f1 = run_few_shot(encoder, tokenizer, dataset["train"], dataset["test"], n, seed)
                scores.append(round(f1, 4))
                print(f"  N={n:3d} | seed={seed} | macro-F1={f1:.4f}")
            all_results[cond][str(n)] = scores
            mean, std = np.mean(scores), np.std(scores)
            print(f"  N={n:3d} | MEAN={mean:.4f} ± {std:.4f}")

    out_path = os.path.join(RESULTS_DIR, "finetuning_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {out_path}")

    # Quick summary table
    print("\n--- Summary (macro-F1 mean ± std) ---")
    print(f"{'Condition':<15}", end="")
    for n in N_LIST:
        print(f"  N={n:<5}", end="")
    print()
    for cond, res in all_results.items():
        print(f"{cond:<15}", end="")
        for n in N_LIST:
            scores = res.get(str(n), [])
            if scores:
                print(f"  {np.mean(scores):.3f}±{np.std(scores):.3f}", end="")
            else:
                print(f"  {'—':>9}", end="")
        print()


if __name__ == "__main__":
    main()
