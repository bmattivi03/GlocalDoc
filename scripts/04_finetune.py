import sys
import os
import json
import random
import torch
import numpy as np
from torch.optim import AdamW
from sklearn.metrics import f1_score
from transformers import RobertaModel, RobertaTokenizerFast, get_cosine_schedule_with_warmup

sys.path.append(".")
from src.data import ECTHR_LABEL_TEXTS, load_ecthr, sample_few_shot
from src.model import (
    AttentionPooling,
    DocumentClassifier,
    GlocalIBModel,
    ProtoClassifier,
)

# --- CONFIG ---
N_LIST          = [10, 50, 100]
SEEDS           = [0, 1, 2, 3, 4]
FINETUNE_EPOCHS = 10
LR              = 2e-5
RESULTS_DIR     = "results"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(RESULTS_DIR, exist_ok=True)

# `no_pretrain` is a sanity baseline: fresh distilroberta-base + fresh attention pool.
# If GlocalIB cannot beat this, the pre-training added nothing.
CONDITIONS = {
    "glocal_ib":   ("checkpoints/glocal_ib_epoch4.pt", "glocal"),
    "h_mlm":       ("checkpoints/h_mlm_epoch3.pt",     "h_mlm"),
    "no_pretrain": (None,                              "raw"),
}

# V2 (P-E-02 / P-J-03): glocal_ib uses ProtoClassifier so the IB head + projector
# are on-path at evaluation. h_mlm and no_pretrain use ProtoClassifier *without*
# the IB chain (mu_head=None, projector=None) — i.e., same prototype + temperature
# head but the encoder output goes directly to attn_pool. Set to False to revert
# any/all conditions to V1's DocumentClassifier with a Linear head.
USE_PROTO_CLASSIFIER = {"glocal_ib": True, "h_mlm": True, "no_pretrain": True}


def seed_everything(seed: int):
    """Seed all RNGs that affect classifier-head init, dropout, and data shuffles."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def calibrate_temperature(clf, val_split) -> float:
    """P-H-02: post-hoc temperature scaling on validation logits.

    Refits the classifier's temperature (Guo et al. 2017 / arxiv:1706.04599)
    to minimize BCE on val without retraining the rest of the model. Only
    works on ProtoClassifier instances; returns None for DocumentClassifier
    (which has no exposed temperature).

    Returns the new log_temperature (also set in-place on clf).
    """
    if not hasattr(clf, "log_temperature"):
        return None
    clf.eval()
    # Collect val cosines (logits / current temperature) once.
    with torch.no_grad():
        cur_temp = clf.log_temperature.exp().item()
        cosines, targets = [], []
        for ex in val_split:
            logits = clf([ex["text"]]) / cur_temp                  # back to raw cosines
            cosines.append(logits)
            t = torch.zeros(1, 10, device=logits.device)
            for lbl in ex["labels"]:
                if lbl < 10:
                    t[0][lbl] = 1.0
            targets.append(t)
        cosines = torch.cat(cosines, dim=0)
        targets = torch.cat(targets, dim=0)

    log_T = torch.nn.Parameter(torch.zeros(1, device=cosines.device))
    opt = torch.optim.LBFGS([log_T], lr=0.05, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            log_T.exp() * cosines, targets
        )
        loss.backward()
        return loss

    opt.step(closure)
    new_log_T = float(log_T.detach().item())
    clf.log_temperature.data = log_T.detach().clone().view(())
    return new_log_T


def load_encoder(path, ckpt_type):
    """Returns (encoder, tokenizer, attn_pool, mu_head, projector).

    P-E-02 / P-J-03: V1 dropped mu_head and projector when loading a GlocalIB
    checkpoint, making the IB latent invisible at evaluation. V2 returns them
    so the ProtoClassifier can put the IB chain back on-path at fine-tune.
    For h_mlm and no_pretrain, mu_head and projector are None — those
    conditions don't have an IB head.
    """
    if ckpt_type == "glocal":
        # Load the full GlocalIB model and return:
        #  - encoder (the body — RobertaModel)
        #  - tokenizer
        #  - attn_pool_student (gradient-trained, not the EMA copy)
        #  - mu_head, projector (so the IB chain is on-path at fine-tune)
        m = GlocalIBModel(device=DEVICE)
        m.load_state_dict(torch.load(path, map_location=DEVICE))
        return m.encoder, m.tokenizer, m.attn_pool_student, m.mu_head, m.projector
    elif ckpt_type == "h_mlm":
        ckpt      = torch.load(path, map_location=DEVICE)
        tokenizer = RobertaTokenizerFast.from_pretrained("distilroberta-base")
        encoder   = RobertaModel.from_pretrained("distilroberta-base", add_pooling_layer=False)
        encoder.load_state_dict(ckpt["encoder_state"])
        encoder   = encoder.to(DEVICE)
        attn_pool = AttentionPooling(dim=768, max_chunks=50).to(DEVICE)
        attn_pool.load_state_dict(ckpt["attn_pool_state"])
        return encoder, tokenizer, attn_pool, None, None
    else:  # "raw" — no pre-training baseline
        tokenizer = RobertaTokenizerFast.from_pretrained("distilroberta-base")
        encoder   = RobertaModel.from_pretrained("distilroberta-base", add_pooling_layer=False).to(DEVICE)
        attn_pool = AttentionPooling(dim=768, max_chunks=50).to(DEVICE)
        return encoder, tokenizer, attn_pool, None, None


def run_few_shot(
    encoder, tokenizer, attn_pool, mu_head, projector,
    train_split, val_split, test_split, n, seed,
    use_proto: bool = True,
):
    # Seed everything BEFORE constructing the classifier so the head init
    # actually depends on `seed`. Without this, the 5 seeds would only vary
    # which few-shot examples are sampled, not the head init.
    seed_everything(seed)

    few_shot  = sample_few_shot(train_split, n, seed)
    if use_proto:
        clf = ProtoClassifier(
            encoder, tokenizer, attn_pool,
            label_texts=ECTHR_LABEL_TEXTS,
            mu_head=mu_head, projector=projector,
            device=DEVICE,
        )
    else:
        clf = DocumentClassifier(encoder, tokenizer, attn_pool=attn_pool, device=DEVICE)
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
    # P-H-02: post-hoc temperature scaling on val before test eval.
    # No-op for DocumentClassifier (no exposed temperature).
    new_log_T = calibrate_temperature(clf, val_split)
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
    print("Loading dataset...")
    dataset     = load_ecthr()
    all_results = {}

    for cond, (path, ckpt_type) in CONDITIONS.items():
        if path is not None and not os.path.exists(path):
            print(f"Skipping {cond} — checkpoint not found at {path}")
            continue

        print(f"\n=== Condition: {cond} ===")
        encoder, tokenizer, attn_pool, mu_head, projector = load_encoder(path, ckpt_type)
        encoder_state_init   = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}
        attn_pool_state_init = {k: v.cpu().clone() for k, v in attn_pool.state_dict().items()}
        mu_head_state_init    = ({k: v.cpu().clone() for k, v in mu_head.state_dict().items()}
                                 if mu_head is not None else None)
        projector_state_init  = ({k: v.cpu().clone() for k, v in projector.state_dict().items()}
                                 if projector is not None else None)
        all_results[cond] = {}
        use_proto = USE_PROTO_CLASSIFIER.get(cond, True)

        for n in N_LIST:
            macro_scores, micro_scores = [], []
            for seed in SEEDS:
                encoder.load_state_dict(encoder_state_init)
                attn_pool.load_state_dict(attn_pool_state_init)
                if mu_head is not None:
                    mu_head.load_state_dict(mu_head_state_init)
                if projector is not None:
                    projector.load_state_dict(projector_state_init)
                metrics = run_few_shot(
                    encoder, tokenizer, attn_pool, mu_head, projector,
                    dataset["train"], dataset["validation"], dataset["test"],
                    n, seed,
                    use_proto=use_proto,
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
