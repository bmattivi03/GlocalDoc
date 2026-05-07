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
from src.model import GlocalIBModel, DocumentClassifier, AttentionPooling

# --- CONFIG ---
N_LIST          = [10, 50, 100]
SEEDS           = [0, 1, 2, 3, 4]
FINETUNE_EPOCHS = 10
LR              = 2e-5
RESULTS_DIR     = "results"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(RESULTS_DIR, exist_ok=True)

# Two conditions: glocal_ib uses checkpoint epoch 4; mlm uses HF saved model.
CONDITIONS = {
    "glocal_ib": ("checkpoints/glocal_ib_epoch4.pt", "glocal"),
    "mlm":       ("checkpoints/mlm_baseline",         "mlm"),
}


def load_encoder(path, ckpt_type):
    """Returns (encoder, tokenizer, attn_pool) ready for DocumentClassifier."""
    if ckpt_type == "glocal":
        m = GlocalIBModel(device=DEVICE)
        state = torch.load(path, map_location=DEVICE)
        m.load_state_dict(state)
        # Teacher attention pool was shaped by full-document inputs throughout pre-training.
        return m.encoder, m.tokenizer, m.attn_pool_teacher
    else:
        mlm = RobertaForMaskedLM.from_pretrained(path)
        tokenizer = RobertaTokenizerFast.from_pretrained("distilroberta-base")
        # MLM has no pre-trained attention pool — use a fresh one (zero-init = mean pooling).
        attn_pool = AttentionPooling(dim=768, max_chunks=50).to(DEVICE)
        return mlm.roberta.to(DEVICE), tokenizer, attn_pool


def run_few_shot(encoder, tokenizer, attn_pool, train_split, val_split, test_split, n, seed):
    few_shot = sample_few_shot(train_split, n, seed)
    clf = DocumentClassifier(encoder, tokenizer, attn_pool=attn_pool, device=DEVICE)
    opt = AdamW(clf.parameters(), lr=LR)
    # DocumentClassifier.forward already applies sigmoid → use BCELoss, not BCEWithLogitsLoss.
    criterion = torch.nn.BCELoss()

    best_f1    = -1.0
    best_state = None

    for epoch in range(FINETUNE_EPOCHS):
        clf.train()
        for ex in few_shot:
            opt.zero_grad()
            probs = clf([ex["text"]])           # (1, 10) — sigmoid probabilities
            labels = torch.zeros(1, 10, device=DEVICE)
            for lbl in ex["labels"]:
                if lbl < 10:
                    labels[0][lbl] = 1.0
            loss = criterion(probs, labels)
            loss.backward()
            opt.step()

        # Validation
        clf.eval()
        v_preds, v_targets = [], []
        with torch.no_grad():
            for ex in val_split:
                probs = clf([ex["text"]])        # (1, 10) — already sigmoid
                p = probs.cpu().numpy()[0]
                v_preds.append((p >= 0.5).astype(int))
                t = np.zeros(10, dtype=int)
                for lbl in ex["labels"]:
                    if lbl < 10:
                        t[lbl] = 1
                v_targets.append(t)

        v_f1 = f1_score(np.array(v_targets), np.array(v_preds), average="macro", zero_division=0)
        if v_f1 > best_f1:
            best_f1    = v_f1
            best_state = {k: v.cpu().clone() for k, v in clf.state_dict().items()}

    clf.load_state_dict(best_state)
    clf.eval()
    preds, targets = [], []
    with torch.no_grad():
        for ex in test_split:
            probs = clf([ex["text"]])
            p = probs.cpu().numpy()[0]
            preds.append((p >= 0.5).astype(int))
            t = np.zeros(10, dtype=int)
            for lbl in ex["labels"]:
                if lbl < 10:
                    t[lbl] = 1
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
        encoder, tokenizer, attn_pool = load_encoder(path, ckpt_type)
        all_results[cond] = {}

        for n in N_LIST:
            scores = []
            for seed in SEEDS:
                f1 = run_few_shot(
                    encoder, tokenizer, attn_pool,
                    dataset["train"], dataset["validation"], dataset["test"],
                    n, seed,
                )
                scores.append(round(f1, 4))
                print(f"  N={n:3d} | seed={seed} | macro-F1={f1:.4f}")
            all_results[cond][str(n)] = scores
            mean, std = np.mean(scores), np.std(scores)
            print(f"  N={n:3d} | MEAN={mean:.4f} ± {std:.4f}")

    out_path = os.path.join(RESULTS_DIR, "finetuning_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {out_path}")

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
