import contextlib
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    DataCollatorForLanguageModeling,
    RobertaForMaskedLM,
    RobertaModel,
    RobertaTokenizerFast,
)


class AttentionPooling(nn.Module):
    def __init__(self, dim: int = 768, max_chunks: int = 50):
        super().__init__()
        self.attn_query = nn.Parameter(torch.randn(dim) * 0.01)
        self.chunk_pos  = nn.Embedding(max_chunks, dim)
        nn.init.zeros_(self.chunk_pos.weight)   # zero-init → starts as mean pooling

    def forward(self, chunk_vecs: torch.Tensor, return_weights: bool = False):
        N       = chunk_vecs.size(0)
        pos_ids = torch.arange(N, device=chunk_vecs.device)
        vecs    = chunk_vecs + self.chunk_pos(pos_ids)
        scores  = vecs @ self.attn_query                    # (N,)
        weights = torch.softmax(scores, dim=0)              # (N,)
        doc_vec = (weights.unsqueeze(-1) * vecs).sum(0)     # (768,)
        if return_weights:
            return doc_vec, weights
        return doc_vec


class GlocalIBModel(nn.Module):
    def __init__(
        self,
        hidden_dim: int     = 256,
        proj_dim: int       = 512,
        max_paragraphs: int = 50,
        ema_tau: float      = 0.996,
        device: str         = "cuda",
    ):
        super().__init__()
        self.tokenizer = RobertaTokenizerFast.from_pretrained("distilroberta-base")
        # P-C-01: student is RobertaForMaskedLM so we have a built-in MLM head
        # for per-paragraph token-prediction. `self.encoder` is a PROPERTY that
        # returns the body — using an attribute assignment would duplicate the
        # body's parameters under both `encoder.*` and `encoder_full.*` names
        # in named_parameters().
        self.encoder_full = RobertaForMaskedLM.from_pretrained("distilroberta-base")
        self.encoder_full.config.use_cache = False
        self.encoder_full.roberta.config.use_cache = False

        # BYOL-style EMA teacher encoder — body only, no MLM head. Provides
        # alignment targets that are stable across student updates. Updated by
        # EMA only, never by gradient.
        self.encoder_teacher = copy.deepcopy(self.encoder_full.roberta)
        for p in self.encoder_teacher.parameters():
            p.requires_grad = False
        self.encoder_teacher.eval()

        # P-C-01: HF data collator does the 15%-mask sampling cleanly.
        self.mlm_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm_probability=0.15, return_tensors="pt"
        )

        # Separate attention pools: student trains via gradient, teacher updated via EMA.
        self.attn_pool_student = AttentionPooling(dim=768, max_chunks=max_paragraphs)
        self.attn_pool_teacher = AttentionPooling(dim=768, max_chunks=max_paragraphs)
        self.attn_pool_teacher.load_state_dict(self.attn_pool_student.state_dict())
        for p in self.attn_pool_teacher.parameters():
            p.requires_grad = False

        # IB probabilistic head (student only)
        self.mu_head        = nn.Linear(768, hidden_dim)
        self.log_sigma_head = nn.Linear(768, hidden_dim)

        # MLP projector: hidden_dim → proj_dim → 768
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, 768),
        )

        # BYOL/SimSiam predictor — applied on the STUDENT side only, to both the
        # IB-projected vector (before L_global) and the partial-pool vector
        # (before L_inter). This asymmetry is the formal mechanism that prevents
        # representation collapse in non-contrastive self-distillation.
        # LayerNorm (not BatchNorm) because batch_size=1.
        self.predictor = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 768),
        )

        # Homoscedastic UW weights: [local, inter, global]. P-G-02 pulled
        # L_compress out of UW because UW assumes likelihoods and KL-to-prior
        # is a regularizer — feeding it to UW attenuated the compression
        # gradient by ~e^(-log(128)) ≈ 0.008 in V1.
        self.log_s = nn.Parameter(torch.zeros(3))

        # P-B-04: DINO-style teacher centering. Running mean of teacher
        # full-doc outputs, subtracted from Z_prime before alignment so the
        # teacher cannot collapse to a single output. EMA momentum 0.9.
        # Accumulation: forward adds the batch-mean (×B) into _pending_center_sum
        # and increments _pending_center_count; update_teacher_ema divides
        # and applies EMA, then resets. Under GRAD_ACCUM > 1 this ensures all
        # micro-batches in the optimizer step contribute, not just the last
        # (the bug a code review caught at glocaldoc-v2:670ae21).
        self.register_buffer("teacher_center", torch.zeros(768))
        self.center_momentum = 0.9
        self.register_buffer("_pending_center_sum", torch.zeros(768))
        self.register_buffer("_pending_center_count", torch.zeros(1))

        self.ema_tau        = ema_tau
        self.max_paragraphs = max_paragraphs
        self.to(device)

    # P-C-01: properties for the encoder body + LM head, to avoid duplicating
    # parameter registrations under multiple module paths.
    @property
    def encoder(self):
        return self.encoder_full.roberta

    @property
    def lm_head(self):
        return self.encoder_full.lm_head

    # ------------------------------------------------------------------
    # Internal paragraph encoding
    # ------------------------------------------------------------------

    def _encode_paragraphs(self, paragraphs: list, use_teacher: bool) -> torch.Tensor:
        """
        Encode a list of paragraph strings → (N, 768).

        Paragraphs longer than 510 tokens are split into non-overlapping sub-chunks,
        each encoded via CLS, then mean-pooled into one vector. All sub-chunks
        across all paragraphs are batched into a single encoder forward pass.

        `use_teacher=True` uses the frozen EMA teacher encoder under torch.no_grad;
        no activations are retained. `use_teacher=False` uses the student encoder
        with gradient.
        """
        CHUNK_SIZE  = 510
        sub_chunks: list = []
        boundaries: list = []

        for para in paragraphs:
            ids   = self.tokenizer.encode(para, add_special_tokens=False)
            start = len(sub_chunks)
            if len(ids) <= CHUNK_SIZE:
                sub_chunks.append(para)
            else:
                for i in range(0, len(ids), CHUNK_SIZE):
                    sub_chunks.append(self.tokenizer.decode(ids[i:i + CHUNK_SIZE]))
            boundaries.append((start, len(sub_chunks)))

        enc_module = self.encoder_teacher if use_teacher else self.encoder
        enc = self.tokenizer(
            sub_chunks, padding=True, truncation=True,
            max_length=512, return_tensors="pt",
        ).to(enc_module.device)

        ctx = torch.no_grad() if use_teacher else contextlib.nullcontext()
        with ctx:
            cls_vecs = enc_module(**enc).last_hidden_state[:, 0, :]

        return torch.stack([cls_vecs[s:e].mean(0) for s, e in boundaries])

    # ------------------------------------------------------------------
    # P-C-01: per-paragraph token MLM loss
    # ------------------------------------------------------------------

    def compute_mlm_loss(self, full_batch: list) -> torch.Tensor:
        """Per-paragraph masked-language-modeling loss across all paragraphs.

        Closes the 30× supervision-density gap that V1's GlocalIB had vs
        H-MLM. Each paragraph in `full_batch` is tokenized, masked at 15%
        via the HF collator, encoded through the STUDENT body, and scored
        through the LM head with cross-entropy against the original tokens
        (only at the masked positions, since the collator sets labels=-100
        elsewhere).

        Returned tensor carries gradient through encoder + lm_head. Caller
        adds it OUTSIDE the UW stack with a fixed coefficient.
        """
        all_para_inputs = []
        for doc in full_batch:
            for para in doc:
                ids = self.tokenizer.encode(para, truncation=True, max_length=512)
                if len(ids) > 2:  # at least one non-special token
                    all_para_inputs.append({"input_ids": ids})
        if not all_para_inputs:
            return self.encoder.embeddings.word_embeddings.weight.new_zeros(())
        batch = self.mlm_collator(all_para_inputs)
        batch = {k: v.to(self.encoder.device) for k, v in batch.items()}
        out = self.encoder_full(**batch)
        return out.loss

    # ------------------------------------------------------------------
    # EMA update (call after every optimizer.step)
    # ------------------------------------------------------------------

    def update_teacher_ema(self):
        """EMA-update teacher encoder, teacher attention pool, and the DINO
        teacher_center (P-B-04). Buffers (LayerNorm running stats, etc.) are
        hard-copied from student to avoid drift."""
        with torch.no_grad():
            # Encoder parameters
            for p_t, p_s in zip(
                self.encoder_teacher.parameters(),
                self.encoder.parameters(),
            ):
                p_t.data.mul_(self.ema_tau).add_(p_s.data, alpha=1.0 - self.ema_tau)
            # Encoder buffers (LayerNorm running mean/var if any) — direct copy
            for b_t, b_s in zip(
                self.encoder_teacher.buffers(),
                self.encoder.buffers(),
            ):
                b_t.data.copy_(b_s.data)
            # Attention pool
            for p_t, p_s in zip(
                self.attn_pool_teacher.parameters(),
                self.attn_pool_student.parameters(),
            ):
                p_t.data.mul_(self.ema_tau).add_(p_s.data, alpha=1.0 - self.ema_tau)
            # P-B-04: teacher centering. Consume the accumulated sum + count
            # across all micro-batches in this optimizer step (GRAD_ACCUM > 1
            # otherwise only the last micro-batch's mean would land in the EMA).
            if self._pending_center_count.item() > 0:
                batch_mean = self._pending_center_sum / self._pending_center_count.item()
                self.teacher_center.mul_(self.center_momentum).add_(
                    batch_mean, alpha=1.0 - self.center_momentum
                )
                self._pending_center_sum.zero_()
                self._pending_center_count.zero_()

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        full_batch:         list,   # list[list[str]] — clean paragraphs (X0)
        masked_batch:       list,   # list[list[str]] — word/sentence masked (Xm)
        kept_indices_batch: list,   # list[list[int]] — M kept indices per doc
    ):
        """
        Returns 10-tuple. Loss code consumes indices 0–7; indices 8–9 are for
        logging only (raw pre-predictor vectors to diagnose predictor health).

          0  Z_prime_centered (B, 768)        teacher full-doc repr with DINO centering applied (stop-grad)
          1  Z_proj_pred     (B, 768)         student post-IB, post-predictor (for L_global)
          2  z_partial_pred  (B, 768)         student partial-pool, post-predictor (for L_inter)
          3  s_chunks        list[(N, 768)]   student all-paragraph reps (for L_local, var, cov)
          4  t_chunks        list[(N, 768)]   teacher all-paragraph reps (for L_local)
          5  mu              (B, 256)         IB mean
          6  log_sigma       (B, 256)         IB log-std (clamped to [-10, 10])
          7  log_s           (3,)             UW weights [local, inter, global] (clamped)
          8  Z_proj          (B, 768)         student post-IB pre-predictor (logging only)
          9  z_partial       (B, 768)         student pre-IB pre-predictor (logging only)
        """
        Z_prime_list        = []
        Z_proj_pred_list    = []
        z_partial_pred_list = []
        Z_proj_list         = []
        z_partial_list      = []
        s_chunks_list       = []
        t_chunks_list       = []
        mu_list             = []
        log_sigma_list      = []

        for full, masked, kept in zip(full_batch, masked_batch, kept_indices_batch):
            # ── Teacher: EMA encoder pass under no_grad, then EMA attention pool
            t_chunks = self._encode_paragraphs(full, use_teacher=True)     # (N, 768)
            with torch.no_grad():
                Z_prime = self.attn_pool_teacher(t_chunks)                 # (768,)

            # ── Student: trainable encoder pass
            s_chunks = self._encode_paragraphs(masked, use_teacher=False)  # (N, 768)

            # Paragraph dropout for the partial-pool view
            valid_kept = [i for i in kept if i < len(s_chunks)]
            if not valid_kept:
                valid_kept = [0]
            s_kept    = s_chunks[valid_kept]                               # (M, 768)
            z_partial = self.attn_pool_student(s_kept)                    # (768,)

            # IB bottleneck — clamp log_sigma to prevent over/underflow under bf16.
            mu        = self.mu_head(z_partial)                            # (256,)
            log_sigma = torch.clamp(self.log_sigma_head(z_partial), -10, 10)
            sigma     = torch.exp(log_sigma)
            z_sample  = mu + sigma * torch.randn_like(mu)
            Z_proj    = self.projector(z_sample)                          # (768,)

            # Predictor: applied on student side only — the asymmetry that
            # breaks the trivial collapse fixed point.
            Z_proj_pred    = self.predictor(Z_proj)
            z_partial_pred = self.predictor(z_partial)

            Z_prime_list.append(Z_prime)
            Z_proj_pred_list.append(Z_proj_pred)
            z_partial_pred_list.append(z_partial_pred)
            Z_proj_list.append(Z_proj)
            z_partial_list.append(z_partial)
            s_chunks_list.append(s_chunks)
            t_chunks_list.append(t_chunks)
            mu_list.append(mu)
            log_sigma_list.append(log_sigma)

        log_s = torch.clamp(self.log_s, min=-10, max=10)

        # P-B-04: DINO-style centering. Accumulate (sum, count) into the
        # pending buffers — every micro-batch in this optimizer step contributes
        # to the EMA that runs in update_teacher_ema(). Subtract the *current*
        # running center (not the post-EMA value) from Z_prime so the teacher
        # cannot trivially collapse to a single output.
        Z_prime_stacked = torch.stack(Z_prime_list)
        with torch.no_grad():
            batch_sum = Z_prime_stacked.detach().sum(dim=0)
            batch_n   = float(Z_prime_stacked.shape[0])
            self._pending_center_sum.add_(batch_sum)
            self._pending_center_count.add_(batch_n)
            Z_prime_centered = Z_prime_stacked - self.teacher_center.unsqueeze(0)

        return (
            Z_prime_centered,                  # 0 (B, 768) — center-subtracted
            torch.stack(Z_proj_pred_list),     # 1 (B, 768)
            torch.stack(z_partial_pred_list),  # 2 (B, 768)
            s_chunks_list,                     # 3 list[(N, 768)]
            t_chunks_list,                     # 4 list[(N, 768)]
            torch.stack(mu_list),              # 5 (B, 256)
            torch.stack(log_sigma_list),       # 6 (B, 256)
            log_s,                             # 7 (3,)
            torch.stack(Z_proj_list),          # 8 (B, 768)  logging only
            torch.stack(z_partial_list),       # 9 (B, 768)  logging only
        )


class DocumentClassifier(nn.Module):
    """
    Fine-tuning classifier. Loads encoder and attention pool from a pre-trained
    checkpoint.

    For GlocalIB: pass `m.attn_pool_student` (the pool that actually trained
    via gradient — `attn_pool_teacher` is an EMA of the student and barely
    diverges from random init over a single short pre-training run).
    For H-MLM: pass the trained `attn_pool`.
    """

    def __init__(
        self,
        encoder,
        tokenizer,
        attn_pool,
        num_labels: int     = 10,
        max_paragraphs: int = 50,
        device: str         = "cuda",
    ):
        super().__init__()
        self.encoder        = encoder
        self.tokenizer      = tokenizer
        self.attn_pool      = attn_pool
        self.classifier     = nn.Linear(768, num_labels)
        self.max_paragraphs = max_paragraphs
        self.to(device)

    def _encode_paragraphs(self, paragraphs: list) -> torch.Tensor:
        """Identical sub-chunk logic to GlocalIBModel._encode_paragraphs (stop_grad=False)."""
        CHUNK_SIZE  = 510
        sub_chunks: list = []
        boundaries: list = []

        for para in paragraphs:
            ids   = self.tokenizer.encode(para, add_special_tokens=False)
            start = len(sub_chunks)
            if len(ids) <= CHUNK_SIZE:
                sub_chunks.append(para)
            else:
                for i in range(0, len(ids), CHUNK_SIZE):
                    sub_chunks.append(self.tokenizer.decode(ids[i:i + CHUNK_SIZE]))
            boundaries.append((start, len(sub_chunks)))

        enc = self.tokenizer(
            sub_chunks, padding=True, truncation=True,
            max_length=512, return_tensors="pt",
        ).to(self.encoder.device)

        cls_vecs = self.encoder(**enc).last_hidden_state[:, 0, :]
        return torch.stack([cls_vecs[s:e].mean(0) for s, e in boundaries])

    def forward(self, paragraphs_batch: list) -> torch.Tensor:
        """Returns raw logits (B, num_labels). Apply sigmoid at eval / use BCEWithLogitsLoss at train."""
        doc_vecs = torch.stack([
            self.attn_pool(self._encode_paragraphs(paras))
            for paras in paragraphs_batch
        ])                                               # (B, 768)
        return self.classifier(doc_vecs)                # (B, num_labels) — logits


class ProtoClassifier(nn.Module):
    """V2 fine-tune classifier (P-E-02 + P-J-03).

    Differences from V1's DocumentClassifier:

    1. Uses the IB head and projector at fine-tune time. V1's load_encoder
       discarded mu_head/log_sigma_head/projector → the IB latent the
       bottleneck was trained to produce was *invisible at evaluation*. V2
       routes the document representation through projector(mu_head(...)),
       making the IB on-path.

    2. Label-text-initialized prototypes (one (768,) vector per class).
       Each prototype starts as projector(mu_head(encoder(label_text))) and
       is then fine-tuned via gradient. Cosine similarity to the prototypes
       gives the logits (scaled by a learnable temperature).

    Pass mu_head=None, projector=None to skip the IB chain (use this for
    h_mlm and no_pretrain so the comparison is apples-to-apples — they
    don't have an IB head).
    """

    def __init__(
        self,
        encoder,
        tokenizer,
        attn_pool,
        label_texts: list,
        mu_head: nn.Module | None = None,
        projector: nn.Module | None = None,
        device: str = "cuda",
        temperature_init: float = 10.0,
        max_paragraphs: int = 50,
    ):
        super().__init__()
        self.encoder        = encoder
        self.tokenizer      = tokenizer
        self.attn_pool      = attn_pool
        self.mu_head        = mu_head
        self.projector      = projector
        self.max_paragraphs = max_paragraphs
        # Learnable temperature — multiplies the cosine before BCE/CE. Init
        # at 10 puts the cosines in [-10, 10] which is roughly the right
        # scale for BCEWithLogitsLoss to receive meaningful gradient.
        self.log_temperature = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(temperature_init)))))

        # Initialize prototypes by encoding label texts through the same
        # representation path the documents will be encoded through.
        with torch.no_grad():
            proto_init = []
            for txt in label_texts:
                enc = tokenizer(
                    txt, return_tensors="pt", truncation=True, max_length=64
                ).to(device)
                # Encoder body output → take CLS → IB head → projector
                cls_vec = encoder(**enc).last_hidden_state[:, 0, :]   # (1, 768)
                if mu_head is not None and projector is not None:
                    rep = projector(mu_head(cls_vec)).squeeze(0)
                else:
                    rep = cls_vec.squeeze(0)
                proto_init.append(rep)
        self.prototypes = nn.Parameter(torch.stack(proto_init))    # (num_labels, 768)
        self.to(device)

    @property
    def num_labels(self) -> int:
        return self.prototypes.shape[0]

    def _doc_rep(self, paragraphs: list) -> torch.Tensor:
        """(768,) document representation. Goes through the IB chain if available."""
        CHUNK_SIZE  = 510
        sub_chunks: list = []
        boundaries: list = []
        for para in paragraphs:
            ids   = self.tokenizer.encode(para, add_special_tokens=False)
            start = len(sub_chunks)
            if len(ids) <= CHUNK_SIZE:
                sub_chunks.append(para)
            else:
                for i in range(0, len(ids), CHUNK_SIZE):
                    sub_chunks.append(self.tokenizer.decode(ids[i:i + CHUNK_SIZE]))
            boundaries.append((start, len(sub_chunks)))
        enc = self.tokenizer(
            sub_chunks, padding=True, truncation=True,
            max_length=512, return_tensors="pt",
        ).to(self.encoder.device)
        cls_vecs = self.encoder(**enc).last_hidden_state[:, 0, :]
        para_reps = torch.stack([cls_vecs[s:e].mean(0) for s, e in boundaries])  # (N, 768)
        partial   = self.attn_pool(para_reps)                                    # (768,)
        if self.mu_head is not None and self.projector is not None:
            # IB head is on-path at fine-tune: μ (deterministic at eval),
            # then projector back to 768.
            partial = self.projector(self.mu_head(partial))
        return partial

    def forward(self, paragraphs_batch: list) -> torch.Tensor:
        """Returns logits (B, num_labels) = temperature · cosine(doc_rep, prototype)."""
        doc_reps = torch.stack([self._doc_rep(paras) for paras in paragraphs_batch])  # (B, 768)
        doc_n    = F.normalize(doc_reps,        dim=-1)
        proto_n  = F.normalize(self.prototypes, dim=-1)
        temp     = self.log_temperature.exp()
        return temp * (doc_n @ proto_n.T)                                              # (B, num_labels)
