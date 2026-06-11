import sys
import os
import json
import random
import argparse
import torch
import numpy as np
from torch.optim import AdamW
from sklearn.metrics import f1_score
from transformers import RobertaModel, RobertaTokenizerFast, get_cosine_schedule_with_warmup

sys.path.append(".")
from src.data import load_ecthr, sample_few_shot
from src.model import GlocalIBModel, DocumentClassifier, AttentionPooling

# --- CONFIG ---
# Defaults below reproduce the finalized runs. Override the N grid, seeds, or output
# path from the CLI (see --help) to run a finer sweep WITHOUT touching the canonical
# results file. Example finer grid:
#   python scripts/04_finetune.py --n-list 5,10,15,20,30,50 \
#       --out results/finetuning_results_finegrid.json
DEFAULT_N_LIST  = [10, 50, 100]
DEFAULT_SEEDS   = [0, 1, 2, 3, 4]
FINETUNE_EPOCHS = 10
LR              = 2e-5
RESULTS_DIR     = "results"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(RESULTS_DIR, exist_ok=True)


def _int_csv(text):
    """Parse a comma-separated list of ints, e.g. '5,10,15' -> [5, 10, 15]."""
    return [int(x) for x in str(text).split(",") if x.strip() != ""]


def build_parser():
    p = argparse.ArgumentParser(
        description="Few-shot fine-tuning across conditions, N, and seeds.")
    p.add_argument("--n-list", dest="n_list", type=_int_csv, default=DEFAULT_N_LIST,
                   help="Comma-separated labelled-examples-per-class grid "
                        "(default: 10,50,100). Finer sweep e.g.: 5,10,15,20,30,50.")
    p.add_argument("--seeds", type=_int_csv, default=DEFAULT_SEEDS,
                   help="Comma-separated seeds (default: 0,1,2,3,4).")
    p.add_argument("--out", default=os.path.join(RESULTS_DIR, "finetuning_results.json"),
                   help="Output JSON (default: results/finetuning_results.json). "
                        "Use a separate file for a new grid to keep finalized results.")
    p.add_argument("--force", action="store_true",
                   help="Allow overwriting --out if it already exists.")
    return p

# `no_pretrain` is a sanity baseline: fresh distilroberta-base + fresh attention pool.
# If GlocalIB cannot beat this, the pre-training added nothing.
CONDITIONS = {
    "glocal_ib":   ("checkpoints/glocal_ib_epoch4.pt", "glocal"),
    "h_mlm":       ("checkpoints/h_mlm_epoch3.pt",     "h_mlm"),
    "no_pretrain": (None,                              "raw"),
}


def seed_everything(seed: int):
    """Seed all RNGs that affect classifier-head init, dropout, and data shuffles."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_encoder(path, ckpt_type):
    """Returns (encoder, tokenizer, attn_pool) ready for DocumentClassifier."""
    if ckpt_type == "glocal":
        # Load the full GlocalIB model and return the STUDENT pool — that's the
        # one that actually trained via gradient. The teacher pool is an EMA of
        # the student and barely diverges from random init over a short pre-train.
        m = GlocalIBModel(device=DEVICE)
        m.load_state_dict(torch.load(path, map_location=DEVICE))
        return m.encoder, m.tokenizer, m.attn_pool_student
    elif ckpt_type == "h_mlm":
        ckpt      = torch.load(path, map_location=DEVICE)
        tokenizer = RobertaTokenizerFast.from_pretrained("distilroberta-base")
        encoder   = RobertaModel.from_pretrained("distilroberta-base", add_pooling_layer=False)
        encoder.load_state_dict(ckpt["encoder_state"])
        encoder   = encoder.to(DEVICE)
        attn_pool = AttentionPooling(dim=768, max_chunks=50).to(DEVICE)
        attn_pool.load_state_dict(ckpt["attn_pool_state"])
        return encoder, tokenizer, attn_pool
    else:  # "raw" — no pre-training baseline
        tokenizer = RobertaTokenizerFast.from_pretrained("distilroberta-base")
        encoder   = RobertaModel.from_pretrained("distilroberta-base", add_pooling_layer=False).to(DEVICE)
        attn_pool = AttentionPooling(dim=768, max_chunks=50).to(DEVICE)
        return encoder, tokenizer, attn_pool


def run_few_shot(encoder, tokenizer, attn_pool, train_split, val_split, test_split, n, seed):
    # Seed everything BEFORE constructing DocumentClassifier so the classifier
    # head's nn.Linear init actually depends on `seed`. Without this, the 5 seeds
    # would only vary which few-shot examples are sampled, not the head init.
    seed_everything(seed)

    few_shot  = sample_few_shot(train_split, n, seed)
    clf       = DocumentClassifier(encoder, tokenizer, attn_pool=attn_pool, device=DEVICE)
    opt       = AdamW(clf.parameters(), lr=LR)
    # BCEWithLogitsLoss — numerically stable with raw logits (DocumentClassifier returns logits).
    criterion = torch.nn.BCEWithLogitsLoss()

    total_steps  = FINETUNE_EPOCHS * len(few_shot)
    warmup_steps = max(1, total_steps // 10)
    scheduler    = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)

    best_macro = -1.0
    best_state = None

    for epoch in range(FINETUNE_EPOCHS):
        clf.train()
        for ex in few_shot:
            opt.zero_grad()
            logits = clf([ex["text"]])           # (1, 10) — raw logits
            labels = torch.zeros(1, 10, device=DEVICE)
            for lbl in ex["labels"]:
                if lbl < 10:
                    labels[0][lbl] = 1.0
            loss = criterion(logits, labels)
            loss.backward()
            opt.step()
            scheduler.step()

        # Validation
        clf.eval()
        v_preds, v_targets = [], []
        with torch.no_grad():
            for ex in val_split:
                logits = clf([ex["text"]])
                p      = torch.sigmoid(logits).cpu().numpy()[0]
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
            logits = clf([ex["text"]])
            p      = torch.sigmoid(logits).cpu().numpy()[0]
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
    args     = build_parser().parse_args()
    n_list   = args.n_list
    seeds    = args.seeds
    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if os.path.exists(out_path) and not args.force:
        sys.exit(f"Refusing to overwrite existing {out_path}. "
                 f"Pass --out <new path> for a fresh grid, or --force to overwrite.")

    print(f"N grid: {n_list} | seeds: {seeds} | out: {out_path}")
    print("Loading dataset...")
    dataset     = load_ecthr()
    all_results = {}

    for cond, (path, ckpt_type) in CONDITIONS.items():
        if path is not None and not os.path.exists(path):
            print(f"Skipping {cond} — checkpoint not found at {path}")
            continue

        print(f"\n=== Condition: {cond} ===")
        encoder, tokenizer, attn_pool = load_encoder(path, ckpt_type)
        encoder_state_init   = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}
        attn_pool_state_init = {k: v.cpu().clone() for k, v in attn_pool.state_dict().items()}
        all_results[cond] = {}

        for n in n_list:
            macro_scores, micro_scores = [], []
            for seed in seeds:
                encoder.load_state_dict(encoder_state_init)
                attn_pool.load_state_dict(attn_pool_state_init)
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

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {out_path}")

    print("\n--- Summary (macro-F1 mean ± std) ---")
    print(f"{'Condition':<15}", end="")
    for n in n_list:
        print(f"  N={n:<5}", end="")
    print()
    for cond, res in all_results.items():
        print(f"{cond:<15}", end="")
        for n in n_list:
            scores = res.get(str(n), {}).get("macro_f1", [])
            if scores:
                print(f"  {np.mean(scores):.3f}±{np.std(scores):.3f}", end="")
            else:
                print(f"  {'—':>9}", end="")
        print()

    print("\n--- Summary (micro-F1 mean ± std) ---")
    print(f"{'Condition':<15}", end="")
    for n in n_list:
        print(f"  N={n:<5}", end="")
    print()
    for cond, res in all_results.items():
        print(f"{cond:<15}", end="")
        for n in n_list:
            scores = res.get(str(n), {}).get("micro_f1", [])
            if scores:
                print(f"  {np.mean(scores):.3f}±{np.std(scores):.3f}", end="")
            else:
                print(f"  {'—':>9}", end="")
        print()


if __name__ == "__main__":
    main()
