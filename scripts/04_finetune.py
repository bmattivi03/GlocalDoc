import sys
import os
import json
import torch
import numpy as np
from torch.optim import AdamW
from sklearn.metrics import f1_score
from transformers import RobertaModel, RobertaTokenizerFast, get_cosine_schedule_with_warmup

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

CONDITIONS = {
    "glocal_ib": ("checkpoints/glocal_ib_epoch4.pt", "glocal"),
    "mlm":       ("checkpoints/h_mlm_epoch3.pt",      "h_mlm"),
}


def load_encoder(path, ckpt_type):
    """Returns (encoder, tokenizer, attn_pool) ready for DocumentClassifier."""
    if ckpt_type == "glocal":
        m = GlocalIBModel(device=DEVICE)
        m.load_state_dict(torch.load(path, map_location=DEVICE))
        return m.encoder, m.tokenizer, m.attn_pool_teacher
    else:  # h_mlm
        ckpt      = torch.load(path, map_location=DEVICE)
        tokenizer = RobertaTokenizerFast.from_pretrained("distilroberta-base")
        encoder   = RobertaModel.from_pretrained("distilroberta-base")
        encoder.load_state_dict(ckpt["encoder_state"])
        encoder   = encoder.to(DEVICE)
        attn_pool = AttentionPooling(dim=768, max_chunks=50).to(DEVICE)
        attn_pool.load_state_dict(ckpt["attn_pool_state"])
        return encoder, tokenizer, attn_pool


def run_few_shot(encoder, tokenizer, attn_pool, train_split, val_split, test_split, n, seed):
    few_shot  = sample_few_shot(train_split, n, seed)
    clf       = DocumentClassifier(encoder, tokenizer, attn_pool=attn_pool, device=DEVICE)
    opt       = AdamW(clf.parameters(), lr=LR)
    criterion = torch.nn.BCELoss()

    total_steps  = FINETUNE_EPOCHS * len(few_shot)
    warmup_steps = max(1, total_steps // 10)
    scheduler    = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)

    best_macro = -1.0
    best_state = None

    for epoch in range(FINETUNE_EPOCHS):
        clf.train()
        for ex in few_shot:
            opt.zero_grad()
            probs  = clf([ex["text"]])           # (1, 10) — sigmoid probabilities
            labels = torch.zeros(1, 10, device=DEVICE)
            for lbl in ex["labels"]:
                if lbl < 10:
                    labels[0][lbl] = 1.0
            loss = criterion(probs, labels)
            loss.backward()
            opt.step()
            scheduler.step()

        # Validation
        clf.eval()
        v_preds, v_targets = [], []
        with torch.no_grad():
            for ex in val_split:
                probs = clf([ex["text"]])
                p     = probs.cpu().numpy()[0]
                v_preds.append((p >= 0.5).astype(int))
                t = np.zeros(10, dtype=int)
                for lbl in ex["labels"]:
                    if lbl < 10:
                        t[lbl] = 1
                v_targets.append(t)

        v_macro = f1_score(np.array(v_targets), np.array(v_preds), average="macro", zero_division=0)
        if v_macro > best_macro:
            best_macro = v_macro
            best_state = {k: v.cpu().clone() for k, v in clf.state_dict().items()}

    clf.load_state_dict(best_state)
    clf.eval()
    preds, targets = [], []
    with torch.no_grad():
        for ex in test_split:
            probs = clf([ex["text"]])
            p     = probs.cpu().numpy()[0]
            preds.append((p >= 0.5).astype(int))
            t = np.zeros(10, dtype=int)
            for lbl in ex["labels"]:
                if lbl < 10:
                    t[lbl] = 1
            targets.append(t)

    macro_f1 = f1_score(np.array(targets), np.array(preds), average="macro", zero_division=0)
    micro_f1 = f1_score(np.array(targets), np.array(preds), average="micro", zero_division=0)
    return {"macro_f1": round(macro_f1, 4), "micro_f1": round(micro_f1, 4)}


def main():
    print("Loading dataset...")
    dataset     = load_ecthr()
    all_results = {}

    for cond, (path, ckpt_type) in CONDITIONS.items():
        if not os.path.exists(path):
            print(f"Skipping {cond} — checkpoint not found at {path}")
            continue

        print(f"\n=== Condition: {cond} ===")
        encoder, tokenizer, attn_pool = load_encoder(path, ckpt_type)
        all_results[cond] = {}

        for n in N_LIST:
            macro_scores, micro_scores = [], []
            for seed in SEEDS:
                metrics = run_few_shot(
                    encoder, tokenizer, attn_pool,
                    dataset["train"], dataset["validation"], dataset["test"],
                    n, seed,
                )
                macro_scores.append(metrics["macro_f1"])
                micro_scores.append(metrics["micro_f1"])
                print(f"  N={n:3d} | seed={seed} | macro-F1={metrics['macro_f1']:.4f} | micro-F1={metrics['micro_f1']:.4f}")

            all_results[cond][str(n)] = {
                "macro_f1": macro_scores,
                "micro_f1": micro_scores,
            }
            print(f"  N={n:3d} | macro MEAN={np.mean(macro_scores):.4f}±{np.std(macro_scores):.4f}"
                  f" | micro MEAN={np.mean(micro_scores):.4f}±{np.std(micro_scores):.4f}")

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
            scores = res.get(str(n), {}).get("macro_f1", [])
            if scores:
                print(f"  {np.mean(scores):.3f}±{np.std(scores):.3f}", end="")
            else:
                print(f"  {'—':>9}", end="")
        print()

    print("\n--- Summary (micro-F1 mean ± std) ---")
    print(f"{'Condition':<15}", end="")
    for n in N_LIST:
        print(f"  N={n:<5}", end="")
    print()
    for cond, res in all_results.items():
        print(f"{cond:<15}", end="")
        for n in N_LIST:
            scores = res.get(str(n), {}).get("micro_f1", [])
            if scores:
                print(f"  {np.mean(scores):.3f}±{np.std(scores):.3f}", end="")
            else:
                print(f"  {'—':>9}", end="")
        print()


if __name__ == "__main__":
    main()
