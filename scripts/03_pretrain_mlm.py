import os
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")  # silence unauth-request warning
# Force NCCL onto shared-memory/socket transports (matches scripts/02). Harmless on
# NVLink hardware and single GPU; protects against PCIe-only P2P deadlocks elsewhere.
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

import sys
import time
from collections import deque
from datetime import timedelta

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import transformers
from transformers import (
    RobertaForMaskedLM, RobertaTokenizerFast,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)
from accelerate import Accelerator, InitProcessGroupKwargs

transformers.logging.set_verbosity_error()

sys.path.append(".")
from src.data import load_ecthr, get_paragraph_mask
from src.model import AttentionPooling
from src.loss import alignment_loss

# ── CONFIG ────────────────────────────────────────────────────────────────────
EPOCHS        = 3
BATCH_SIZE    = 1       # per GPU; 1 doc per step (para count varies)
GRAD_ACCUM    = 4
LR            = 5e-5
MAX_GRAD_NORM = 1.0
ALPHA         = 0.5     # weight on L_mlm; (1-ALPHA) on L_para_pred
LOG_EVERY     = 1       # W&B log frequency (optimizer steps)
PRINT_EVERY   = 50      # stdout summary frequency (optimizer steps)
WANDB_PROJECT = "glocal-nlp"
CONDITION     = "h_mlm"
# ─────────────────────────────────────────────────────────────────────────────


def collate_fn(batch):
    return [item["text"] for item in batch]


class HMLMTrainer(nn.Module):
    """Single nn.Module wrapping RobertaForMaskedLM + AttentionPooling.

    Both the L_mlm encoder pass and the L_para_pred encoder passes happen inside
    this forward, so when accelerator.prepare() wraps it under DDP:
      - all gradient flows from both losses go through one DDP forward call,
      - mixed-precision autocast applies uniformly to all encoder calls,
      - gradient checkpointing applies uniformly.
    """

    def __init__(self, encoder_mlm: RobertaForMaskedLM, attn_pool: AttentionPooling):
        super().__init__()
        self.encoder_mlm = encoder_mlm
        self.attn_pool   = attn_pool

    def forward(
        self,
        mlm_input_ids,        # (N_para, L) MLM-corrupted ids
        mlm_attention_mask,   # (N_para, L)
        mlm_labels,           # (N_para, L)
        full_input_ids,       # (N_para, L) clean paragraph ids (for teacher)
        full_attention_mask,  # (N_para, L)
        kept_input_ids,       # (M, L) clean kept-paragraph ids (for student pool)
        kept_attention_mask,  # (M, L)
    ):
        # ── L_mlm: per-paragraph masked language modelling ──
        l_mlm = self.encoder_mlm(
            input_ids=mlm_input_ids,
            attention_mask=mlm_attention_mask,
            labels=mlm_labels,
        ).loss

        # ── L_para_pred: stop-grad full-doc mean vs student kept-doc pool ──
        with torch.no_grad():
            t_cls = self.encoder_mlm.roberta(
                input_ids=full_input_ids,
                attention_mask=full_attention_mask,
            ).last_hidden_state[:, 0, :]            # (N, 768)
        target = t_cls.mean(0)                       # (768,)

        s_cls = self.encoder_mlm.roberta(
            input_ids=kept_input_ids,
            attention_mask=kept_attention_mask,
        ).last_hidden_state[:, 0, :]                # (M, 768)
        z_partial = self.attn_pool(s_cls)            # (768,)

        l_para = alignment_loss(z_partial.unsqueeze(0), target.detach().unsqueeze(0))
        return l_mlm, l_para


def train():
    # 30-min collective timeout — slow first step under JIT shouldn't kill the run.
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=GRAD_ACCUM,
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(minutes=30))],
    )
    device = accelerator.device

    if accelerator.is_main_process:
        print(
            f"[launch] world_size={accelerator.num_processes} "
            f"mixed_precision={accelerator.mixed_precision} "
            f"device={device}",
            flush=True,
        )
        os.makedirs("checkpoints", exist_ok=True)
        import wandb
        wandb.init(project=WANDB_PROJECT, name=CONDITION, config={
            "condition":  CONDITION,
            "epochs":     EPOCHS,
            "batch_size": BATCH_SIZE,
            "grad_accum": GRAD_ACCUM,
            "world_size": accelerator.num_processes,
            "lr":         LR,
            "alpha":      ALPHA,
        })
    accelerator.wait_for_everyone()

    tokenizer = RobertaTokenizerFast.from_pretrained("distilroberta-base")
    encoder_mlm = RobertaForMaskedLM.from_pretrained("distilroberta-base")
    encoder_mlm.config.use_cache = False
    attn_pool = AttentionPooling(dim=768, max_chunks=50)

    trainer = HMLMTrainer(encoder_mlm, attn_pool)

    mlm_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm_probability=0.15, return_tensors="pt"
    )

    dataset = load_ecthr()
    loader  = DataLoader(
        dataset["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )

    total_steps  = (len(loader) // GRAD_ACCUM) * EPOCHS
    warmup_steps = max(1, total_steps // 10)
    opt          = AdamW(trainer.parameters(), lr=LR)
    scheduler    = get_linear_schedule_with_warmup(opt, warmup_steps, total_steps)

    trainer, opt, loader, scheduler = accelerator.prepare(
        trainer, opt, loader, scheduler
    )

    if accelerator.is_main_process:
        print(
            f"[plan] total_steps={total_steps} warmup_steps={warmup_steps} "
            f"len(loader)={len(loader)}",
            flush=True,
        )

    global_step   = 0
    step_times    = deque(maxlen=50)
    t_train_start = time.time()

    for epoch in range(EPOCHS):
        trainer.train()

        loader_iter = loader
        pbar = None
        if accelerator.is_main_process:
            pbar = tqdm(
                loader,
                desc=f"epoch {epoch + 1}/{EPOCHS}",
                leave=False,
                dynamic_ncols=True,
            )
            loader_iter = pbar

        opt_step_start = time.time()

        for doc_batch in loader_iter:
            paragraphs = doc_batch[0]   # BATCH_SIZE=1 → single document

            # MLM-masked paragraphs
            para_ids  = [
                {"input_ids": tokenizer.encode(p, truncation=True, max_length=512)}
                for p in paragraphs
            ]
            mlm_batch = {k: v.to(device) for k, v in mlm_collator(para_ids).items()}

            # Clean full-doc tokens (teacher input)
            full_enc = tokenizer(
                paragraphs, padding=True, truncation=True,
                max_length=512, return_tensors="pt",
            ).to(device)

            # Paragraph dropout — kept-paragraph tokens (student pool input)
            kept_idx   = get_paragraph_mask(len(paragraphs))
            kept_paras = [paragraphs[i] for i in kept_idx]
            kept_enc   = tokenizer(
                kept_paras, padding=True, truncation=True,
                max_length=512, return_tensors="pt",
            ).to(device)

            with accelerator.accumulate(trainer):
                l_mlm, l_para = trainer(
                    mlm_input_ids       = mlm_batch["input_ids"],
                    mlm_attention_mask  = mlm_batch["attention_mask"],
                    mlm_labels          = mlm_batch["labels"],
                    full_input_ids      = full_enc["input_ids"],
                    full_attention_mask = full_enc["attention_mask"],
                    kept_input_ids      = kept_enc["input_ids"],
                    kept_attention_mask = kept_enc["attention_mask"],
                )
                total = ALPHA * l_mlm + (1.0 - ALPHA) * l_para
                accelerator.backward(total)

                grad_norm = None
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(trainer.parameters(), MAX_GRAD_NORM)

                opt.step()
                scheduler.step()
                opt.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                step_dt = time.time() - opt_step_start
                opt_step_start = time.time()
                step_times.append(step_dt)
                avg_step = sum(step_times) / len(step_times)
                eta_sec  = avg_step * max(0, total_steps - global_step)

                if accelerator.is_main_process:
                    if torch.cuda.is_available():
                        gpu_mem_gb = torch.cuda.max_memory_allocated() / 1e9
                        torch.cuda.reset_peak_memory_stats()
                    else:
                        gpu_mem_gb = 0.0

                    grad_norm_value = grad_norm.item() if grad_norm is not None else 0.0
                    ppl = float(torch.exp(l_mlm.detach()).clamp(max=1e6))

                    if global_step % LOG_EVERY == 0:
                        import wandb
                        wandb.log({
                            "total_loss":       total.item(),
                            "l_mlm":            l_mlm.item(),
                            "l_para":           l_para.item(),
                            "mlm_perplexity":   ppl,
                            "lr":               scheduler.get_last_lr()[0],
                            "grad_norm":        grad_norm_value,
                            "sec_per_step":     avg_step,
                            "examples_per_sec": (BATCH_SIZE * GRAD_ACCUM * accelerator.num_processes) / max(avg_step, 1e-9),
                            "gpu_mem_gb":       gpu_mem_gb,
                            "eta_min":          eta_sec / 60.0,
                            "elapsed_min":      (time.time() - t_train_start) / 60.0,
                            "epoch":            epoch,
                            "progress":         global_step / max(1, total_steps),
                            "step":             global_step,
                        })

                    if pbar is not None:
                        pbar.set_postfix({
                            "loss": f"{total.item():.3f}",
                            "mlm":  f"{l_mlm.item():.3f}",
                            "ppl":  f"{ppl:.1f}",
                            "para": f"{l_para.item():.3f}",
                        })

                    if global_step % PRINT_EVERY == 0:
                        print(
                            f"[step {global_step}/{total_steps}] "
                            f"loss={total.item():.3f}  "
                            f"l_mlm={l_mlm.item():.3f} (ppl={ppl:.1f})  "
                            f"l_para={l_para.item():.3f}  "
                            f"grad={grad_norm_value:.2f} mem={gpu_mem_gb:.1f}GB  "
                            f"sec/step={avg_step:.2f} eta={eta_sec/60:.1f}min",
                            flush=True,
                        )

        if pbar is not None:
            pbar.close()

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            elapsed_min = (time.time() - t_train_start) / 60.0
            print(f"Epoch {epoch} done.  elapsed={elapsed_min:.1f}min", flush=True)
            unwrapped = accelerator.unwrap_model(trainer)
            torch.save(
                {
                    "encoder_state":   unwrapped.encoder_mlm.roberta.state_dict(),
                    "attn_pool_state": unwrapped.attn_pool.state_dict(),
                },
                f"checkpoints/{CONDITION}_epoch{epoch + 1}.pt",
            )
            print(f"  Saved → checkpoints/{CONDITION}_epoch{epoch + 1}.pt", flush=True)

    if accelerator.is_main_process:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    train()
