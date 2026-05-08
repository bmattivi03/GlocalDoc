import os
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    RobertaForMaskedLM, RobertaTokenizerFast,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)
from accelerate import Accelerator
import sys

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


def encode_paragraphs(model, tokenizer, paragraphs, stop_grad):
    """Encode list[str] → (N, 768) CLS vectors via a single batched forward pass."""
    device = model.roberta.embeddings.word_embeddings.weight.device
    enc = tokenizer(
        paragraphs, padding=True, truncation=True,
        max_length=512, return_tensors="pt",
    ).to(device)
    if stop_grad:
        with torch.no_grad():
            return model.roberta(**enc).last_hidden_state[:, 0, :]
    return model.roberta(**enc).last_hidden_state[:, 0, :]


def compute_para_pred_loss(model, tokenizer, attn_pool, paragraphs):
    """L_para_pred: cosine dist between partial-doc pool and stop-grad full-doc mean."""
    t_cls  = encode_paragraphs(model, tokenizer, paragraphs, stop_grad=True)   # (N, 768)
    target = t_cls.mean(0)                                                      # (768,)

    kept_idx   = get_paragraph_mask(len(paragraphs))
    kept_paras = [paragraphs[i] for i in kept_idx]
    s_cls      = encode_paragraphs(model, tokenizer, kept_paras, stop_grad=False)  # (M, 768)
    z_partial  = attn_pool(s_cls)                                                   # (768,)

    return alignment_loss(z_partial.unsqueeze(0), target.detach().unsqueeze(0))


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

    tokenizer   = RobertaTokenizerFast.from_pretrained("distilroberta-base")
    model       = RobertaForMaskedLM.from_pretrained("distilroberta-base")
    model.gradient_checkpointing_enable()
    attn_pool   = AttentionPooling(dim=768, max_chunks=50)
    mlm_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm_probability=0.15, return_tensors="pt"
    )

    dataset = load_ecthr()
    loader  = DataLoader(
        dataset["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )

    total_steps  = (len(loader) // GRAD_ACCUM) * EPOCHS
    warmup_steps = max(1, total_steps // 10)
    opt          = AdamW(list(model.parameters()) + list(attn_pool.parameters()), lr=LR)
    scheduler    = get_linear_schedule_with_warmup(opt, warmup_steps, total_steps)

    model, attn_pool, opt, loader, scheduler = accelerator.prepare(
        model, attn_pool, opt, loader, scheduler
    )

    global_step = 0
    for epoch in range(EPOCHS):
        model.train()
        attn_pool.train()

        for doc_batch in loader:
            paragraphs = doc_batch[0]   # BATCH_SIZE=1 → single document

            with accelerator.accumulate(model, attn_pool):
                # === L_mlm: per-paragraph 15% token masking ===
                para_ids  = [
                    {"input_ids": tokenizer.encode(p, truncation=True, max_length=512)}
                    for p in paragraphs
                ]
                mlm_batch = {k: v.to(device) for k, v in mlm_collator(para_ids).items()}
                l_mlm     = model(**mlm_batch).loss

                # === L_para_pred: partial → full document alignment ===
                # unwrap_model used only for .roberta attribute access (teacher + student encoder)
                # attn_pool passed as DDP-wrapped so its gradients are all-reduced correctly
                l_para = compute_para_pred_loss(
                    accelerator.unwrap_model(model),
                    tokenizer,
                    attn_pool,
                    paragraphs,
                )

                total = ALPHA * l_mlm + (1.0 - ALPHA) * l_para
                accelerator.backward(total)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        list(accelerator.unwrap_model(model).parameters())
                        + list(accelerator.unwrap_model(attn_pool).parameters()),
                        MAX_GRAD_NORM,
                    )
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

        if accelerator.is_main_process:
            print(f"Epoch {epoch} done.")
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(
                {
                    "encoder_state":   accelerator.unwrap_model(model).roberta.state_dict(),
                    "attn_pool_state": accelerator.unwrap_model(attn_pool).state_dict(),
                },
                f"checkpoints/{CONDITION}_epoch{epoch + 1}.pt",
            )
            print(f"  Saved → checkpoints/{CONDITION}_epoch{epoch + 1}.pt")

    if accelerator.is_main_process:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    train()
