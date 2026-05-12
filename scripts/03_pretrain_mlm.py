import os
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")  # silence unauth-request warning

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import transformers
from transformers import (
    RobertaForMaskedLM, RobertaTokenizerFast,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)
from accelerate import Accelerator
import sys

transformers.logging.set_verbosity_error()

sys.path.append(".")
from src.data import load_ecthr, get_paragraph_mask
from src.model import AttentionPooling
from src.loss import alignment_loss

# --- CONFIG ---
EPOCHS        = 3
BATCH_SIZE    = 1       # per GPU; 1 doc per step (para count varies)
GRAD_ACCUM    = 4
LR            = 5e-5
MAX_GRAD_NORM = 1.0
ALPHA         = 0.5     # weight on L_mlm; (1-ALPHA) on L_para_pred
WANDB_PROJECT = "glocal-nlp"
CONDITION     = "h_mlm"


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
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=GRAD_ACCUM,
    )
    device = accelerator.device

    if accelerator.is_main_process:
        import wandb
        wandb.init(project=WANDB_PROJECT, name=CONDITION, config={
            "condition":  CONDITION,
            "epochs":     EPOCHS,
            "batch_size": BATCH_SIZE,
            "grad_accum": GRAD_ACCUM,
            "lr":         LR,
            "alpha":      ALPHA,
        })

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

    global_step = 0
    for epoch in range(EPOCHS):
        trainer.train()

        for doc_batch in loader:
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
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainer.parameters(), MAX_GRAD_NORM)
                opt.step()
                scheduler.step()
                opt.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                if global_step % 50 == 0 and accelerator.is_main_process:
                    import wandb
                    wandb.log({
                        "total_loss": total.item(),
                        "l_mlm":      l_mlm.item(),
                        "l_para":     l_para.item(),
                        "lr":         scheduler.get_last_lr()[0],
                        "epoch":      epoch,
                        "step":       global_step,
                    })

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            print(f"Epoch {epoch} done.")
            os.makedirs("checkpoints", exist_ok=True)
            unwrapped = accelerator.unwrap_model(trainer)
            torch.save(
                {
                    "encoder_state":   unwrapped.encoder_mlm.roberta.state_dict(),
                    "attn_pool_state": unwrapped.attn_pool.state_dict(),
                },
                f"checkpoints/{CONDITION}_epoch{epoch + 1}.pt",
            )
            print(f"  Saved → checkpoints/{CONDITION}_epoch{epoch + 1}.pt")

    if accelerator.is_main_process:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    train()
