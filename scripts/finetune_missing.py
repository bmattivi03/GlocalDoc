#!/usr/bin/env python
"""TEMPORARY / THROWAWAY — fill in only the fine-tuning N values you are MISSING.

You already have results/finetuning_results.json for N in {10, 50, 100}. This script
runs ONLY the N values that are not there yet (the finer grid around the crossover,
e.g. {5, 15, 20, 30}) so you do NOT re-run the slow N=50/100 jobs. It then merges the
new runs with your existing results into one combined file ready for plotting.

It reuses the exact, tested functions from scripts/04_finetune.py (same load_encoder,
run_few_shot, per-seed weight snapshot/restore), so the new numbers are directly
comparable to the ones you already shipped.

Run on the GPU box, from the repo root:
    conda activate glocal_nlp
    python scripts/finetune_missing.py

Outputs:
    results/finetuning_results_finegrid.json   <- only the newly-run N
    results/finetuning_results_combined.json   <- existing + new, sorted by N (for plots)

Your finalized results/finetuning_results.json is never touched. Delete this file when
you are done — it is a throwaway helper, not part of the pipeline.
"""
import os
import sys
import json
import importlib.util

import numpy as np

# --- resolve paths and make checkpoint/relative paths work from anywhere ---
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
os.chdir(ROOT)                 # so "checkpoints/..." and "results/..." resolve
sys.path.append(ROOT)

# Load the real fine-tune module (filename starts with a digit -> importlib).
_spec = importlib.util.spec_from_file_location("ft", os.path.join(HERE, "04_finetune.py"))
ft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ft)

# --- the grid you ultimately want for the macro-F1 vs N plot ---
DESIRED  = [5, 10, 15, 20, 30, 50, 100]
SEEDS    = ft.DEFAULT_SEEDS                       # [0, 1, 2, 3, 4]

EXISTING = os.path.join("results", "finetuning_results.json")
FINEGRID = os.path.join("results", "finetuning_results_finegrid.json")
COMBINED = os.path.join("results", "finetuning_results_combined.json")


def _missing_N():
    """N in DESIRED that are NOT already computed for every condition in EXISTING."""
    existing = json.load(open(EXISTING)) if os.path.exists(EXISTING) else {}
    conds = [c for c in existing if existing[c]]
    if conds:
        already = set.intersection(*[{int(k) for k in existing[c]} for c in conds])
    else:
        already = set()
    return [n for n in DESIRED if n not in already], sorted(already)


def run(missing_N):
    print(f"Filling in missing N={missing_N}, seeds={SEEDS}")
    print("Loading dataset...")
    dataset = ft.load_ecthr()
    results = {}

    for cond, (path, ckpt_type) in ft.CONDITIONS.items():
        if path is not None and not os.path.exists(path):
            print(f"Skipping {cond} — checkpoint not found at {path}")
            continue

        print(f"\n=== Condition: {cond} ===")
        encoder, tokenizer, attn_pool = ft.load_encoder(path, ckpt_type)
        # Snapshot pre-trained weights once, restore before every run so all
        # seed x N runs start identical (same invariant as 04_finetune.py).
        enc_init  = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}
        pool_init = {k: v.cpu().clone() for k, v in attn_pool.state_dict().items()}
        results[cond] = {}

        for n in missing_N:
            macro, micro = [], []
            for seed in SEEDS:
                encoder.load_state_dict(enc_init)
                attn_pool.load_state_dict(pool_init)
                m = ft.run_few_shot(
                    encoder, tokenizer, attn_pool,
                    dataset["train"], dataset["validation"], dataset["test"],
                    n, seed,
                )
                macro.append(m["macro_f1"])
                micro.append(m["micro_f1"])
                print(f"  N={n:3d} | seed={seed} | macro-F1={m['macro_f1']:.4f} | micro-F1={m['micro_f1']:.4f}")
            results[cond][str(n)] = {"macro_f1": macro, "micro_f1": micro}
            print(f"  N={n:3d} | macro MEAN={np.mean(macro):.4f}±{np.std(macro):.4f}"
                  f" | micro MEAN={np.mean(micro):.4f}±{np.std(micro):.4f}")

    with open(FINEGRID, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nNewly-run results → {FINEGRID}")
    return results


def merge():
    base = json.load(open(EXISTING)) if os.path.exists(EXISTING) else {}
    fine = json.load(open(FINEGRID)) if os.path.exists(FINEGRID) else {}
    combined = {}
    for cond in set(base) | set(fine):
        merged = {}
        merged.update(base.get(cond, {}))
        merged.update(fine.get(cond, {}))           # new N never overlap existing
        combined[cond] = {k: merged[k] for k in sorted(merged, key=int)}
    with open(COMBINED, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"Combined grid → {COMBINED}")

    print("\n--- Combined macro-F1 (mean ± std) ---")
    all_N = sorted({int(k) for c in combined.values() for k in c}, key=int)
    print(f"{'Condition':<14}" + "".join(f"  N={n:<6}" for n in all_N))
    for cond, res in combined.items():
        row = f"{cond:<14}"
        for n in all_N:
            s = res.get(str(n), {}).get("macro_f1", [])
            row += f"  {np.mean(s):.3f}±{np.std(s):.3f}" if s else f"  {'—':>9}"
        print(row)


if __name__ == "__main__":
    missing_N, already = _missing_N()
    print(f"Already computed (skipped): {already}")
    if not missing_N:
        print("Nothing missing — all DESIRED N already present. Just merging.")
    else:
        run(missing_N)
    merge()
