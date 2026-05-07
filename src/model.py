import contextlib
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
        ema_tau: float      = 0.99,
        device: str         = "cuda",
    ):
        super().__init__()
        self.tokenizer = RobertaTokenizerFast.from_pretrained("distilroberta-base")
        # Single shared encoder. Teacher pass uses stop-grad; student pass trains it.
        self.encoder   = RobertaModel.from_pretrained("distilroberta-base")
        self.encoder.gradient_checkpointing_enable()

        # Separate attention pools: student trains via gradient, teacher updated via EMA.
        self.attn_pool_student = AttentionPooling(dim=768, max_chunks=max_paragraphs)
        self.attn_pool_teacher = AttentionPooling(dim=768, max_chunks=max_paragraphs)
        # Teacher attention is never updated by gradient — EMA only.
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

        # Homoscedastic UW weights: [compress, local, inter, global]
        self.log_s = nn.Parameter(torch.zeros(4))

        self.ema_tau        = ema_tau
        self.max_paragraphs = max_paragraphs
        self.to(device)

    # ------------------------------------------------------------------
    # Internal paragraph encoding
    # ------------------------------------------------------------------

    def _encode_paragraphs(self, paragraphs: list, stop_grad: bool) -> torch.Tensor:
        """
        Encode a list of paragraph strings → (N, 768).

        Paragraphs longer than 510 tokens are split into non-overlapping sub-chunks
        of 510 tokens, each encoded via CLS, then mean-pooled into one vector.
        All sub-chunks across all paragraphs are batched into a single encoder
        forward pass, so peak VRAM = one pass regardless of paragraph count.

        Teacher calls use stop_grad=True (torch.no_grad) so no activations are
        retained for backprop, keeping peak VRAM to one student-pass equivalent.
        """
        CHUNK_SIZE  = 510   # leave room for [CLS] and [SEP]
        sub_chunks: list    = []
        boundaries: list    = []   # (start, end) index into sub_chunks per paragraph

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

        ctx = torch.no_grad() if stop_grad else contextlib.nullcontext()
        with ctx:
            cls_vecs = self.encoder(**enc).last_hidden_state[:, 0, :]  # (total_chunks, 768)

        return torch.stack([cls_vecs[s:e].mean(0) for s, e in boundaries])  # (N, 768)

    # ------------------------------------------------------------------
    # EMA update (call after every optimizer.step)
    # ------------------------------------------------------------------

    def update_teacher_ema(self):
        with torch.no_grad():
            for p_t, p_s in zip(
                self.attn_pool_teacher.parameters(),
                self.attn_pool_student.parameters(),
            ):
                p_t.data = self.ema_tau * p_t.data + (1 - self.ema_tau) * p_s.data

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
        Returns 8-tuple:
          Z_prime    (B, 768)        teacher full-doc repr (stop-grad)
          Z_proj     (B, 768)        student post-IB projection
          z_partial  (B, 768)        student pre-IB partial-doc repr (for L_inter)
          s_chunks   list[(N, 768)] student all-paragraph reps (for L_local)
          t_chunks   list[(N, 768)] teacher all-paragraph reps (for L_local)
          mu         (B, 256)        IB mean
          sigma      (B, 256)        IB std
          log_s      (4,)            UW weights (clamped to [-10, 10])
        """
        Z_prime_list   = []
        Z_proj_list    = []
        z_partial_list = []
        s_chunks_list  = []
        t_chunks_list  = []
        mu_list        = []
        sigma_list     = []

        for full, masked, kept in zip(full_batch, masked_batch, kept_indices_batch):
            # ── Teacher: stop-grad encoder pass, then teacher attention pool ──
            t_chunks = self._encode_paragraphs(full, stop_grad=True)       # (N, 768)
            with torch.no_grad():
                Z_prime = self.attn_pool_teacher(t_chunks)                 # (768,)

            # ── Student: full grad encoder pass ──────────────────────────────
            s_chunks = self._encode_paragraphs(masked, stop_grad=False)    # (N, 768)

            # Paragraph dropout: keep M reps for student pooling
            valid_kept = [i for i in kept if i < len(s_chunks)]
            if not valid_kept:
                valid_kept = [0]
            s_kept    = s_chunks[valid_kept]                               # (M, 768)
            z_partial = self.attn_pool_student(s_kept)                    # (768,)

            # IB bottleneck
            mu       = self.mu_head(z_partial)                             # (256,)
            sigma    = torch.exp(self.log_sigma_head(z_partial))           # (256,)
            z_sample = mu + sigma * torch.randn_like(mu)
            Z_proj   = self.projector(z_sample)                           # (768,)

            Z_prime_list.append(Z_prime)
            Z_proj_list.append(Z_proj)
            z_partial_list.append(z_partial)
            s_chunks_list.append(s_chunks)
            t_chunks_list.append(t_chunks)
            mu_list.append(mu)
            sigma_list.append(sigma)

        log_s = torch.clamp(self.log_s, min=-10, max=10)

        return (
            torch.stack(Z_prime_list),    # (B, 768)
            torch.stack(Z_proj_list),     # (B, 768)
            torch.stack(z_partial_list),  # (B, 768)
            s_chunks_list,                # list[Tensor(N, 768)]
            t_chunks_list,                # list[Tensor(N, 768)]
            torch.stack(mu_list),         # (B, 256)
            torch.stack(sigma_list),      # (B, 256)
            log_s,                        # (4,)
        )


class DocumentClassifier(nn.Module):
    """
    Fine-tuning classifier. Loads encoder and teacher attention pool from a
    pre-trained GlocalIBModel checkpoint.

    Usage:
        pretrained = GlocalIBModel(...)
        # load state dict ...
        classifier = DocumentClassifier(
            encoder=pretrained.encoder,
            tokenizer=pretrained.tokenizer,
            attn_pool=pretrained.attn_pool_teacher,
        )
    """

    def __init__(
        self,
        encoder,
        tokenizer,
        attn_pool,                      # pass pretrained.attn_pool_teacher
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
        doc_vecs = torch.stack([
            self.attn_pool(self._encode_paragraphs(paras))
            for paras in paragraphs_batch
        ])                                               # (B, 768)
        return torch.sigmoid(self.classifier(doc_vecs)) # (B, num_labels)
