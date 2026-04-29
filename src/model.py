import contextlib
import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizerFast


class AttentionPooling(nn.Module):
    def __init__(self, dim: int = 768, max_chunks: int = 200):
        super().__init__()
        self.attn_query = nn.Parameter(torch.randn(dim) * 0.01)
        self.chunk_pos  = nn.Embedding(max_chunks, dim)
        nn.init.zeros_(self.chunk_pos.weight)

    def forward(self, chunk_vecs: torch.Tensor, return_weights: bool = False):
        N       = chunk_vecs.size(0)
        pos_ids = torch.arange(N, device=chunk_vecs.device)
        vecs    = chunk_vecs + self.chunk_pos(pos_ids)
        scores  = vecs @ self.attn_query
        weights = torch.softmax(scores, dim=0)
        doc_vec = (weights.unsqueeze(-1) * vecs).sum(0)
        if return_weights:
            return doc_vec, weights
        return doc_vec


class GlocalIBModel(nn.Module):
    def __init__(self, hidden_dim: int = 256, proj_dim: int = 512,
                 max_chunks: int = 50, device: str = "cuda"):
        super().__init__()
        self.tokenizer = RobertaTokenizerFast.from_pretrained("distilroberta-base")
        self.encoder   = RobertaModel.from_pretrained("distilroberta-base")
        self.encoder.gradient_checkpointing_enable()

        self.attention_pool = AttentionPooling(dim=768, max_chunks=200)

        # IB probabilistic head
        self.mu_head        = nn.Linear(768, hidden_dim)
        self.log_sigma_head = nn.Linear(768, hidden_dim)

        # MLP projector: 256 → 512 → 768
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, 768),
        )

        # Homoscedastic uncertainty weights: [compress, local, inter, global]
        self.log_s = nn.Parameter(torch.zeros(4))

        self.max_chunks = max_chunks
        self.to(device)

    def _encode_chunks(self, paragraphs: list, stop_grad: bool) -> torch.Tensor:
        if len(paragraphs) > self.max_chunks:
            half = self.max_chunks // 2
            paragraphs = paragraphs[:half] + paragraphs[-half:]

        enc = self.tokenizer(
            paragraphs, padding=True, truncation=True,
            max_length=512, return_tensors="pt",
        ).to(self.encoder.device)

        ctx = torch.no_grad() if stop_grad else contextlib.nullcontext()
        with ctx:
            out = self.encoder(**enc)

        return out.last_hidden_state[:, 0, :]   # (N, 768)

    def forward(self, full_batch: list, masked_batch: list, kept_indices_batch: list):
        Z_prime_list, Z_proj_list  = [], []
        Z_inter_s_list, Z_inter_t_list = [], []
        chunks_s_list, chunks_t_list   = [], []
        mu_list, sigma_list = [], []

        for full, masked, indices in zip(full_batch, masked_batch, kept_indices_batch):
            # Teacher branch (stop-gradient)
            t_chunks = self._encode_chunks(full, stop_grad=True)          # (N, 768)
            Z_prime  = self.attention_pool(t_chunks)                      # (768,)
            valid_idx = [i for i in indices if i < len(t_chunks)]
            if not valid_idx:
                valid_idx = [0]
            t_chunks_kept = t_chunks[valid_idx]                           # (M, 768)

            # Student branch
            s_chunks  = self._encode_chunks(masked, stop_grad=False)      # (M, 768)
            z_partial = self.attention_pool(s_chunks)                     # (768,)

            # IB bottleneck
            mu       = self.mu_head(z_partial)
            sigma    = torch.exp(self.log_sigma_head(z_partial))
            z_sample = mu + sigma * torch.randn_like(mu)
            Z_proj   = self.projector(z_sample)                           # (768,)

            # Teacher partial pool (for intermediate loss)
            Z_teacher_partial = self.attention_pool(t_chunks_kept)        # (768,)

            Z_prime_list.append(Z_prime)
            Z_proj_list.append(Z_proj)
            Z_inter_s_list.append(z_partial)
            Z_inter_t_list.append(Z_teacher_partial)
            chunks_s_list.append(s_chunks)
            chunks_t_list.append(t_chunks_kept)
            mu_list.append(mu)
            sigma_list.append(sigma)

        return (
            torch.stack(Z_prime_list),       # (B, 768) — teacher full doc
            torch.stack(Z_proj_list),        # (B, 768) — student IB projection
            torch.stack(Z_inter_s_list),     # (B, 768) — student partial aggregate
            torch.stack(Z_inter_t_list),     # (B, 768) — teacher partial aggregate
            chunks_s_list,                   # list[Tensor(M, 768)] — variable length
            chunks_t_list,                   # list[Tensor(M, 768)] — variable length
            torch.stack(mu_list),            # (B, 256)
            torch.stack(sigma_list),         # (B, 256)
            self.log_s,                      # (4,) — learnable weights
        )


class DocumentClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, tokenizer,
                 num_labels: int = 10, max_chunks: int = 50, device: str = "cuda"):
        super().__init__()
        self.encoder     = encoder
        self.tokenizer   = tokenizer
        self.attn_pool   = AttentionPooling(dim=768, max_chunks=200)
        self.classifier  = nn.Linear(768, num_labels)
        self.max_chunks  = max_chunks
        self.to(device)

    def _encode_chunks(self, paragraphs: list) -> torch.Tensor:
        if len(paragraphs) > self.max_chunks:
            half = self.max_chunks // 2
            paragraphs = paragraphs[:half] + paragraphs[-half:]
        enc = self.tokenizer(
            paragraphs, padding=True, truncation=True,
            max_length=512, return_tensors="pt",
        ).to(self.encoder.device)
        out = self.encoder(**enc)
        return out.last_hidden_state[:, 0, :]   # (N, 768)

    def forward(self, paragraphs_batch: list) -> torch.Tensor:
        doc_vecs = torch.stack([
            self.attn_pool(self._encode_chunks(paras))
            for paras in paragraphs_batch
        ])   # (B, 768)
        return torch.sigmoid(self.classifier(doc_vecs))   # (B, num_labels)
