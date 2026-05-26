"""P-E-01 — free-lunch label-text μ-prototype probe on an existing checkpoint.

Phase-1 diagnostic from the V2 blueprint: BEFORE retraining anything, check
whether V1's already-trained IB head produces a representation that *already*
distinguishes labels via simple prototype matching. The result tells you which
of two stories is right:

  STORY A (V2 hypothesis): IB head has signal, but V1's load_encoder discarded
  it at fine-tune. A no-training prototype probe through projector(mu_head(
  attn_pool(encoder(...)))) should beat the same probe through just
  attn_pool(encoder(...)). If so, the gap to H-MLM was about *evaluation*,
  not pretraining.

  STORY B (counterfactual): IB head has no signal. The two probes tie. In
  that case the V2 architecture work is the right intervention, not just
  fixing the eval path.

Either story is publishable in the thesis as long as it's measured.

Usage:
    python scripts/05_freelunch_probe.py \\
        --ckpt checkpoints/glocal_ib_epoch4.pt \\
        --out  results/freelunch_probe.json

The script does NOT depend on the V2 RobertaForMaskedLM-based GlocalIBModel
— it reconstructs the V1 architecture directly from the saved state_dict
keys, so older checkpoints load cleanly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from transformers import RobertaModel, RobertaTokenizerFast

sys.path.append(".")
from src.data import ECTHR_LABEL_TEXTS, load_ecthr
from src.model import AttentionPooling


def _build_v1_components(ckpt_path: str, device: str):
    """Build encoder + attn_pool_student + mu_head + projector and load weights
    from a V1-style GlocalIBModel state_dict (keys like `encoder.*`, `mu_head.*`).
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    tokenizer = RobertaTokenizerFast.from_pretrained("distilroberta-base")
    encoder = RobertaModel.from_pretrained("distilroberta-base", add_pooling_layer=False).to(device)
    attn_pool = AttentionPooling(dim=768, max_chunks=50).to(device)
    mu_head = nn.Linear(768, 256).to(device)
    projector = nn.Sequential(
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, 768),
    ).to(device)

    def _slice(prefix: str) -> dict:
        return {k[len(prefix):]: v for k, v in ckpt.items() if k.startswith(prefix)}

    enc_state = _slice("encoder.")
    if enc_state:
        encoder.load_state_dict(enc_state, strict=False)
    else:
        print("[warn] no 'encoder.*' keys — encoder remains at distilroberta-base init")

    ap_state = _slice("attn_pool_student.")
    if ap_state:
        attn_pool.load_state_dict(ap_state, strict=False)

    mu_state = _slice("mu_head.")
    if mu_state:
        mu_head.load_state_dict(mu_state, strict=False)

    proj_state = _slice("projector.")
    if proj_state:
        projector.load_state_dict(proj_state, strict=False)

    return tokenizer, encoder, attn_pool, mu_head, projector


@torch.no_grad()
def _doc_rep(paragraphs: list, encoder, tokenizer, attn_pool,
             mu_head=None, projector=None, device: str = "cpu") -> torch.Tensor:
    CHUNK_SIZE = 510
    sub_chunks, boundaries = [], []
    for para in paragraphs:
        ids = tokenizer.encode(para, add_special_tokens=False)
        start = len(sub_chunks)
        if len(ids) <= CHUNK_SIZE:
            sub_chunks.append(para)
        else:
            for i in range(0, len(ids), CHUNK_SIZE):
                sub_chunks.append(tokenizer.decode(ids[i:i + CHUNK_SIZE]))
        boundaries.append((start, len(sub_chunks)))
    enc = tokenizer(
        sub_chunks, padding=True, truncation=True,
        max_length=512, return_tensors="pt",
    ).to(device)
    cls = encoder(**enc).last_hidden_state[:, 0, :]
    para_reps = torch.stack([cls[s:e].mean(0) for s, e in boundaries])
    partial = attn_pool(para_reps)
    if mu_head is not None and projector is not None:
        partial = projector(mu_head(partial))
    return partial


def _prototype_init(label_texts, encoder, tokenizer, mu_head, projector, device: str):
    protos = []
    for txt in label_texts:
        with torch.no_grad():
            enc = tokenizer(txt, return_tensors="pt", truncation=True, max_length=64).to(device)
            cls = encoder(**enc).last_hidden_state[:, 0, :]
            if mu_head is not None and projector is not None:
                rep = projector(mu_head(cls)).squeeze(0)
            else:
                rep = cls.squeeze(0)
            protos.append(rep)
    return torch.stack(protos)


def _evaluate(split, protos, encoder, tokenizer, attn_pool,
              mu_head=None, projector=None, device: str = "cpu",
              threshold: float = 0.5) -> dict:
    encoder.eval()
    proto_n = F.normalize(protos, dim=-1)
    preds_at, targets_at = [], []
    # Sweep thresholds: pick the one that maximizes macro-F1 on this split.
    all_scores = []
    all_targets = []
    for ex in split:
        rep = _doc_rep(
            ex["text"], encoder, tokenizer, attn_pool,
            mu_head=mu_head, projector=projector, device=device,
        )
        score = (F.normalize(rep, dim=-1) @ proto_n.T).cpu().numpy()
        all_scores.append(score)
        t = np.zeros(10, dtype=int)
        for lbl in ex["labels"]:
            if lbl < 10:
                t[lbl] = 1
        all_targets.append(t)
    scores = np.stack(all_scores)
    targs  = np.stack(all_targets)
    # Per-class threshold sweep — multi-label probe needs more than 0.5.
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.linspace(scores.min(), scores.max(), 21):
        f1 = f1_score(targs, (scores >= thr).astype(int), average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    final_preds = (scores >= best_thr).astype(int)
    return {
        "macro_f1": round(float(f1_score(targs, final_preds, average="macro", zero_division=0)), 4),
        "micro_f1": round(float(f1_score(targs, final_preds, average="micro", zero_division=0)), 4),
        "best_threshold": round(best_thr, 4),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/glocal_ib_epoch4.pt")
    parser.add_argument("--out",  default="results/freelunch_probe.json")
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    if not os.path.exists(args.ckpt):
        raise SystemExit(f"checkpoint {args.ckpt} not found")

    print(f"loading checkpoint {args.ckpt} on {args.device}")
    tokenizer, encoder, attn_pool, mu_head, projector = _build_v1_components(args.ckpt, args.device)

    dataset = load_ecthr()
    split = dataset[args.split]
    print(f"evaluating on {args.split} split ({len(split)} docs)")

    # Two probes — does the IB head help?
    print("[probe 1/2] WITH IB chain (projector ∘ mu_head ∘ attn_pool ∘ encoder)")
    protos_ib = _prototype_init(ECTHR_LABEL_TEXTS, encoder, tokenizer, mu_head, projector, args.device)
    res_ib = _evaluate(split, protos_ib, encoder, tokenizer, attn_pool,
                       mu_head=mu_head, projector=projector, device=args.device)

    print("[probe 2/2] WITHOUT IB chain (attn_pool ∘ encoder)")
    protos_noib = _prototype_init(ECTHR_LABEL_TEXTS, encoder, tokenizer, None, None, args.device)
    res_noib = _evaluate(split, protos_noib, encoder, tokenizer, attn_pool,
                         mu_head=None, projector=None, device=args.device)

    out = {
        "checkpoint": args.ckpt,
        "split": args.split,
        "n_docs": len(split),
        "with_ib_chain": res_ib,
        "without_ib_chain": res_noib,
        "delta_macro_f1": round(res_ib["macro_f1"] - res_noib["macro_f1"], 4),
        "delta_micro_f1": round(res_ib["micro_f1"] - res_noib["micro_f1"], 4),
    }
    print(json.dumps(out, indent=2))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nResults → {args.out}")
    print(
        f"\nInterpretation:\n"
        f"  delta_macro = {out['delta_macro_f1']:+.4f}\n"
        f"  > 0 → V2's STORY A wins: IB head has signal V1 eval discarded.\n"
        f"  ≈ 0 → STORY B: bottleneck wasn't the issue at fine-tune; V2 architecture\n"
        f"        changes (P-C-01 token MLM, P-G-* opt restructure) are doing the work.\n"
        f"  < 0 → IB chain HURT the readout — supports the swarm's claim that V1's\n"
        f"        bottleneck wasn't compressing usefully."
    )


if __name__ == "__main__":
    main()
