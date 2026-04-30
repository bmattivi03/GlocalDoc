# GlocalDoc v2 — Task List

> See `doc/PLAN.md` for full implementation details, code, and architecture spec.
> Phases must be completed in order. Within a phase, tasks can be parallelised.
>
> **Implementation note (2026-04-29):** `doc/PLAN.md` specifies `DistilRobertaModel` but no such
> class exists in HuggingFace `transformers`. `distilroberta-base` is a 6-layer RoBERTa and loads
> via `RobertaModel.from_pretrained("distilroberta-base")`. All files have been updated accordingly.

---

## Phase 0 — Prerequisites

- [ ] **P0.1** Confirm conda env is set up: `conda activate glocal_nlp && python -c "import torch; print(torch.cuda.is_available())"`
  - _Note: environment `glocal_nlp` may not exist yet — create it per `doc/PLAN.md` § Environment Setup, or use an existing env with the same packages._
- [x] **P0.2** Verify `distilroberta-base` downloads correctly: `python -c "from transformers import RobertaModel; RobertaModel.from_pretrained('distilroberta-base'); print('OK')"`
- [x] **P0.3** Deleted stale W&B debug artifacts from `notebooks/wandb/`

---

## Phase 1 — Core Module Rewrite ✅

**Goal:** `src/model.py` and `src/loss.py` match the v2 architecture spec in `PLAN.md`.
`src/data.py` is unchanged.

### T1.1 — Rewrite `src/model.py` ✅

- [x] Add `AttentionPooling` class (learnable query + positional embeddings, zero-init so training starts equivalent to mean pooling)
- [x] Add `_encode_chunks(paragraphs, stop_grad)` — batched tokenize with `padding=True`, single forward pass, returns `(N, 768)` CLS tensor
- [x] Replace `GlocalIBModel.__init__`: swap `RobertaModel` + `distilroberta-base` (was `roberta-base`), `RobertaTokenizerFast`, add `self.attention_pool`, replace old learnable scalars with `self.log_s = nn.Parameter(torch.zeros(4))`
- [x] Rewrite `GlocalIBModel.forward(full_batch, masked_batch, kept_indices_batch)` to return `(Z_prime, Z_proj, Z_inter_s, Z_inter_t, chunks_s, chunks_t, mu, sigma, log_s)`
- [x] Update `DocumentClassifier`: batched encode + `AttentionPooling`, same return interface

**Acceptance:** ✅ Shape verification script passes — `Z_prime=(1,768)`, `Z_proj=(1,768)`, `mu=(1,256)`, `log_s=(4,)`

### T1.2 — Rewrite `src/loss.py` ✅

- [x] Keep `alignment_loss` and `compression_loss` (signatures unchanged)
- [x] Replace `glocal_ib_loss` with 4-component version accepting `(Z_prime, Z_proj, Z_inter_s, Z_inter_t, chunks_s, chunks_t, mu, sigma, log_s, disable_ib)`, returns `(total, l_compress, l_local, l_inter, l_global)`
- [x] Implement homoscedastic UW weighting: `total = (losses * exp(-log_s) + log_s).sum()`
- [x] Implement `disable_ib=True` path: return `l_global` only, other components are zero tensors

**Acceptance:** ✅ Backward-pass verification passes — `total=36.56`, all five components finite.

### T1.3 — Full verification ✅

- [x] Import check: `from src.model import GlocalIBModel; from src.loss import glocal_ib_loss` — OK
- [x] Forward pass shape check — OK
- [x] Backward pass check — loss finite, grad flows, optimizer step completes
- [ ] 4. Smoke test on lab machine (32GB GPU): `python train_glocal.py` (run ~5 steps, confirm no CUDA OOM)

---

## Phase 2 — Training Script Updates ✅

**Goal:** `train_glocal.py` and `train_mlm.py` use the new model API.

### T2.1 — Update `train_glocal.py` ✅

- [x] `Accelerator(mixed_precision="bf16", gradient_accumulation_steps=4)`
- [x] `BATCH_SIZE = 1`
- [x] Update forward call to unpack new return tuple
- [x] Wrap forward + loss in `with accelerator.accumulate(model):`
- [x] `accelerator.clip_grad_norm_(model.parameters(), 1.0)` inside sync block
- [x] W&B logging: `l_compress`, `l_local`, `l_inter`, `l_global`, all four `log_s[0..3]`
- [x] `CONDITION` config flag at top (`"glocal_ib"` or `"glocal_beta0"`) setting `DISABLE_IB`

### T2.2 — Update `train_mlm.py` ✅

- [x] `RobertaTokenizerFast.from_pretrained("distilroberta-base")` (was `RobertaTokenizer` + `roberta-base`)
- [x] `RobertaForMaskedLM.from_pretrained("distilroberta-base")` (was `roberta-base`)

### T2.3 — Smoke tests (requires GPU)

- [ ] Single-GPU run: `python train_glocal.py` (run 5 steps, check loss logs)
- [ ] Multi-GPU run (if available): `accelerate launch --num_processes=2 train_glocal.py`
- [ ] Confirm `log_s` values appear in W&B

---

## Phase 3 — Pre-training Runs (requires Spark cluster)

**Goal:** Produce three pre-trained checkpoints, one per experimental condition.

### T3.1 — GlocalIB pre-training (`glocal_ib`)

- [ ] Set `CONDITION = "glocal_ib"` in `train_glocal.py`
- [ ] Submit to Spark cluster: `sbatch slurm/pretrain_glocal.sh`
- [ ] Monitor W&B: loss should decrease, all four `log_s` should diverge over training
- [ ] **Diagnostic check:** If all `log_s` stay near 0 by epoch 2, flag as degenerate weighting
- [ ] Verify checkpoint saved at `checkpoints/glocal_ib_epoch4/`

### T3.2 — GlocalIB ablation (`glocal_beta0`)

- [ ] Set `CONDITION = "glocal_beta0"` in `train_glocal.py`
- [ ] Submit: `sbatch slurm/pretrain_glocal.sh`
- [ ] Verify only `l_global` is non-zero in W&B logs

### T3.3 — MLM baseline

- [ ] Submit: `sbatch slurm/pretrain_mlm.sh`
- [ ] Verify MLM loss decreasing in W&B
- [ ] Confirm checkpoint at `checkpoints/mlm_baseline/`

---

## Phase 4 — Notebooks

**Goal:** All five notebooks run end-to-end without errors.

### T4.1 — `01_data_exploration.ipynb`

- [ ] Check paragraph-length histogram, label distribution, masking sanity check
- [ ] Run: `jupyter nbconvert --to notebook --execute notebooks/01_data_exploration.ipynb --ExecutePreprocessor.timeout=300`

### T4.2 — `02_pretrain_glocal.ipynb` ✅

- [x] Updated to use new `GlocalIBModel` API (new return tuple, new loss signature, `accelerator.accumulate`)
- [x] Config cell: `CONDITION`, `DISABLE_IB`, `BATCH_SIZE=1`, `GRAD_ACCUM=4`, `LR`
- [x] W&B logging: all 5 loss components + 4 `log_s` values
- [ ] Local sanity run (2 examples, 1 epoch) before full cluster run

### T4.3 — `03_pretrain_mlm.ipynb` ✅

- [x] Swapped to `"distilroberta-base"` for tokenizer and model
- [ ] Local sanity run with `max_steps=2`

### T4.4 — `04_finetune.ipynb`

- [ ] Load encoder from each of the three pre-trained checkpoints
- [ ] `DocumentClassifier` uses v2 API (batched encode + attention pool)
- [ ] Run all 3 conditions × N ∈ {10, 50, 100} × 5 seeds
- [ ] Save `results/finetuning_results.json`

### T4.5 — `05_evaluate.ipynb`

- [ ] Macro-F1 table: mean ± std per condition per N
- [ ] Performance vs N curve (log x-axis, error bars)
- [ ] `log_s` trajectory fetched from W&B for `glocal_ib` run
- [ ] Ablation delta table: GlocalIB full vs β=0 at each N

---

## Phase 5 — SLURM Scripts ✅

- [x] **T5.1** `slurm/pretrain_glocal.sh` — Spark, 4 GPUs, `accelerate launch --num_processes=4`
- [x] **T5.2** `slurm/pretrain_mlm.sh` — Spark, 4 GPUs, `accelerate launch --num_processes=4`
- [x] **T5.3** `slurm/finetune.sh` — standard node, 1 GPU, `jupyter nbconvert --execute`
- [ ] **T5.4** Update `cd /path/to/GlocalDoc` in all three scripts with actual cluster path
- [ ] **T5.5** Confirm `spark` partition name with lab admin before first submission

---

## Verification (Final)

Run before considering pre-training complete:

- [x] `python -c "from src.model import GlocalIBModel; from src.loss import glocal_ib_loss; print('OK')"` passes
- [x] Single forward pass shapes correct: `Z_prime=(B,768)`, `Z_proj=(B,768)`, `mu=(B,256)`, `log_s=(4,)`
- [x] Loss finite and non-NaN after step 0
- [ ] No CUDA OOM on lab machine (32GB) at `BATCH_SIZE=1`
- [ ] All four `log_s` values logged to W&B at each step
- [ ] Three checkpoint directories exist after pre-training: `glocal_ib/`, `glocal_beta0/`, `mlm_baseline/`
- [ ] `finetuning_results.json` contains all 3 conditions × 3 N values × 5 seeds
- [ ] GlocalIB full macro-F1 at N=10 > MLM baseline macro-F1 at N=10 (primary hypothesis)
- [ ] GlocalIB full macro-F1 at N=10 > GlocalIB β=0 macro-F1 at N=10 (ablation confirms IB term)

---

## Compute Reference

| Job | Node | SLURM flags | Batch size |
|-----|------|-------------|-----------|
| GlocalIB pre-training | Spark 128GB | `--partition=spark --gres=gpu:4` | 1 per GPU (effective 4 via grad accum) |
| MLM pre-training | Spark 128GB | `--partition=spark --gres=gpu:4` | 4 per GPU |
| Fine-tuning | Standard 32GB | `--gres=gpu:1` | 4 |
| Sanity checks / exploration | Standard 32GB | `--gres=gpu:1` | 1 |
