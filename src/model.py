import contextlib
import copy
import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizerFast


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
        # Student encoder — receives gradient.
        self.encoder   = RobertaModel.from_pretrained("distilroberta-base")
        self.encoder.config.use_cache = False

        # BYOL-style EMA teacher encoder (separate deepcopy). The teacher provides
        # alignment targets that are stable across student updates, removing the
        # trivial collapse fixed point where a single shared encoder could satisfy
        # all alignment losses by becoming input-invariant. Updated by EMA only,
        # never by gradient.
        self.encoder_teacher = copy.deepcopy(self.encoder)
        for p in self.encoder_teacher.parameters():
            p.requires_grad = False
        self.encoder_teacher.eval()

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
        # teacher cannot collapse to a single output (which would let the
        # student trivially satisfy L_global by becoming constant). EMA
        # momentum 0.9 — fast enough to track distribution drift, slow
        # enough to smooth per-batch noise.
        self.register_buffer("teacher_center", torch.zeros(768))
        self.center_momentum = 0.9
        self._pending_center_update = None   # set inside forward, consumed in update_teacher_ema

        self.ema_tau        = ema_tau
        self.max_paragraphs = max_paragraphs
        self.to(device)

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
            # P-B-04: teacher centering. Consume the pending batch-mean
            # cached during forward; apply EMA with center_momentum.
            if self._pending_center_update is not None:
                self.teacher_center.mul_(self.center_momentum).add_(
                    self._pending_center_update, alpha=1.0 - self.center_momentum
                )
                self._pending_center_update = None

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

        # P-B-04: DINO-style centering. Cache batch mean for the next EMA update;
        # subtract running center from Z_prime so the teacher cannot trivially
        # collapse to a single output. No gradient through the center.
        Z_prime_stacked = torch.stack(Z_prime_list)
        with torch.no_grad():
            self._pending_center_update = Z_prime_stacked.mean(dim=0).detach()
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
