import os
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import (
    RobertaForMaskedLM, 
    RobertaTokenizerFast, 
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup
)
import sys

sys.path.append(".")
from src.data import load_ecthr, truncate_paragraphs

# --- CONFIG ---
EPOCHS = 3
BATCH_SIZE = 4
LR = 5e-5
WARMUP_STEPS = 100
MAX_LENGTH = 512
DOC_MAX_CHUNKS = 50 # Match Glocal truncation for fairness

def train():
    accelerator = Accelerator(log_with="wandb")
    if accelerator.is_main_process:
        import wandb
        wandb.init(project="glocal-nlp", name="mlm_baseline_robust")

    tokenizer = RobertaTokenizerFast.from_pretrained("distilroberta-base")
    model = RobertaForMaskedLM.from_pretrained("distilroberta-base")
    
    # FIX: Use the same loading function to ensure same data distribution (min_paragraphs filter)
    dataset = load_ecthr()
    
    def tokenize_function(examples):
        # Apply same truncation as Glocal
        # FIX: truncate_paragraphs returns (paragraphs, indices)
        truncated_results = [truncate_paragraphs(doc, max_chunks=DOC_MAX_CHUNKS) for doc in examples["text"]]
        
        # Flatten into segments of MAX_LENGTH
        all_segments = []
        for paras, _ in truncated_results:
            full_text = " ".join(paras)
            # FIX: Tokenize without special tokens, chunk, then add special tokens manually
            # This ensures every chunk has a [CLS] token for the classifier to use.
            tokens = tokenizer.encode(full_text, add_special_tokens=False, truncation=False)
            # Leave room for CLS and SEP tokens
            for i in range(0, len(tokens), MAX_LENGTH - 2):
                segment = tokens[i : i + MAX_LENGTH - 2]
                if len(segment) > 10: # avoid tiny segments
                    segment = [tokenizer.cls_token_id] + segment + [tokenizer.sep_token_id]
                    all_segments.append(segment)
        
        return {"input_ids": all_segments}

    # Remove all columns because we are changing the number of rows
    tokenized_datasets = dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing documents into segments"
    )
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    
    train_loader = DataLoader(tokenized_datasets["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=data_collator)
    optimizer = AdamW(model.parameters(), lr=LR)

    # Scheduler
    num_training_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=num_training_steps
    )

    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    global_step = 0
    for epoch in range(EPOCHS):
        model.train()
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            
            if i % 50 == 0 and accelerator.is_main_process:
                import wandb
                wandb.log({
                    "mlm_loss": loss.item(), 
                    "lr": scheduler.get_last_lr()[0],
                    "epoch": epoch, 
                    "step": global_step
                })
            global_step += 1

    if accelerator.is_main_process:
        model.save_pretrained("checkpoints/mlm_baseline")
        print("MLM baseline saved.")

if __name__ == "__main__":
    train()
