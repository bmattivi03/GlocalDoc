import copy
import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizerFast
from src.data import truncate_paragraphs


class AttentionPooling(nn.Module):
    def __init__(self, dim: int = 768, max_chunks: int = 512):
        super().__init__()
        # FIX: Scaling initialization by dim**-0.5 to keep attention scores in a reasonable range
        self.attn_query = nn.Parameter(torch.randn(dim) * (dim**-0.5))
        self.chunk_pos  = nn.Embedding(max_chunks, dim)
        nn.init.normal_(self.chunk_pos.weight, std=0.02)

    def forward(self, chunk_vecs: torch.Tensor, indices: torch.Tensor = None, return_weights: bool = False):
        """
        Args:
            chunk_vecs: (N, dim) or (B, N, dim)
            indices: (N,) or (B, N)
        """
        if chunk_vecs.dim() == 2:
            N = chunk_vecs.size(0)
            if indices is None:
                indices = torch.arange(N, device=chunk_vecs.device)
            indices = torch.clamp(indices, max=self.chunk_pos.num_embeddings - 1)
            
            vecs    = chunk_vecs + self.chunk_pos(indices)
            scores  = vecs @ self.attn_query
            weights = torch.softmax(scores, dim=0)
            doc_vec = (weights.unsqueeze(-1) * vecs).sum(0)
        else:
            B, N, D = chunk_vecs.shape
            if indices is None:
                indices = torch.arange(N, device=chunk_vecs.device).expand(B, N)
            indices = torch.clamp(indices, max=self.chunk_pos.num_embeddings - 1)
            
            vecs    = chunk_vecs + self.chunk_pos(indices)
            scores  = torch.matmul(vecs, self.attn_query)
            weights = torch.softmax(scores, dim=1)
            doc_vec = (weights.unsqueeze(-1) * vecs).sum(1)

        if return_weights:
            return doc_vec, weights
        return doc_vec


class GlocalIBModel(nn.Module):
    def __init__(self, hidden_dim: int = 256, proj_dim: int = 512,
                 max_chunks: int = 512, device: str = "cuda", ema_decay: float = 0.99):
        super().__init__()
        self.tokenizer = RobertaTokenizerFast.from_pretrained("distilroberta-base")
        
        # Student Backbone
        self.encoder = RobertaModel.from_pretrained("distilroberta-base")
        
        # FIX: Full Teacher Branch (Backbone + Pool) for Momentum Encoder (prevent collapse)
        self.teacher_encoder = copy.deepcopy(self.encoder)
        self.teacher_attention_pool = AttentionPooling(dim=768, max_chunks=max_chunks)
        
        # Now enable checkpointing for student only
        self.encoder.gradient_checkpointing_enable()
        self.attention_pool = AttentionPooling(dim=768, max_chunks=max_chunks)
        
        # Detach teacher params
        for param in self.teacher_encoder.parameters():
            param.requires_grad = False
        for param in self.teacher_attention_pool.parameters():
            param.requires_grad = False

        # IB probabilistic head
        self.mu_head        = nn.Linear(768, hidden_dim)
        self.log_sigma_head = nn.Linear(768, hidden_dim)
        
        # Teacher heads (must be frozen for DDP)
        self.teacher_mu_head = copy.deepcopy(self.mu_head)
        for param in self.teacher_mu_head.parameters():
            param.requires_grad = False

        # MLP projector for global alignment
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, 768),
        )
        self.teacher_projector = copy.deepcopy(self.projector)
        for param in self.teacher_projector.parameters():
            param.requires_grad = False

        # FIX: Predictor heads for local/inter/global alignments to prevent collapse (BYOL-style)
        self.local_predictor = nn.Sequential(
            nn.Linear(768, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, 768),
        )
        self.inter_predictor = nn.Sequential(
            nn.Linear(768, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, 768),
        )
        self.global_predictor = nn.Sequential(
            nn.Linear(768, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, 768),
        )

        # Uncertainty weighting for alignment losses (excluding compression)
        self.log_s = nn.Parameter(torch.zeros(3))
        self.ema_decay = ema_decay
        self.to(device)

    @torch.no_grad()
    def update_teacher(self):
        """EMA update of the teacher's weights."""
        for s_param, t_param in zip(self.encoder.parameters(), self.teacher_encoder.parameters()):
            t_param.data.mul_(self.ema_decay).add_(s_param.data, alpha=1 - self.ema_decay)
        for s_param, t_param in zip(self.attention_pool.parameters(), self.teacher_attention_pool.parameters()):
            t_param.data.mul_(self.ema_decay).add_(s_param.data, alpha=1 - self.ema_decay)
        for s_param, t_param in zip(self.mu_head.parameters(), self.teacher_mu_head.parameters()):
            t_param.data.mul_(self.ema_decay).add_(s_param.data, alpha=1 - self.ema_decay)
        for s_param, t_param in zip(self.projector.parameters(), self.teacher_projector.parameters()):
            t_param.data.mul_(self.ema_decay).add_(s_param.data, alpha=1 - self.ema_decay)

    def _encode_paragraphs(self, paragraphs: list, use_teacher: bool = False, chunk_size: int = 16) -> torch.Tensor:
        """Encodes paragraphs with mini-batching to prevent OOM."""
        if not paragraphs:
            return torch.empty(0, 768, device=self.encoder.device)
            
        encoder = self.teacher_encoder if use_teacher else self.encoder
        all_vecs = []
        
        for i in range(0, len(paragraphs), chunk_size):
            batch = paragraphs[i : i + chunk_size]
            enc = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=256, return_tensors="pt",
            ).to(self.encoder.device)
            out = encoder(**enc)
            all_vecs.append(out.last_hidden_state[:, 0, :])
            
        return torch.cat(all_vecs, dim=0)

    def forward(self, full_batch_data: list, masked_batch: list, kept_indices_batch: list):
        """
        full_batch_data: list of (paragraphs, indices)
        """
        # Ensure teacher is in eval mode to disable dropout
        self.teacher_encoder.eval()
        self.teacher_attention_pool.eval()
        self.teacher_projector.eval()

        # 1. Teacher Pass: Full documents
        full_batch_texts = [d[0] for d in full_batch_data]
        full_batch_indices = [d[1] for d in full_batch_data]
        
        doc_lengths_t = [len(d) for d in full_batch_texts]
        all_full_paras = [p for d in full_batch_texts for p in d]
        
        with torch.no_grad():
            all_vecs_t = self._encode_paragraphs(all_full_paras, use_teacher=True)
            teacher_chunks_list = torch.split(all_vecs_t, doc_lengths_t)

        # 2. Student Pass: Masked documents
        doc_lengths_s = [len(d) for d in masked_batch]
        all_masked_paras = [p for d in masked_batch for p in d]
        all_vecs_s = self._encode_paragraphs(all_masked_paras, use_teacher=False)
        student_chunks_list = torch.split(all_vecs_s, doc_lengths_s)
        
        Z_prime_list, Z_proj_list = [], []
        Z_inter_s_list, Z_inter_t_list = [], []
        chunks_s_list, chunks_t_list = [], []
        mu_list, log_sigma_list = [], []

        for i, (s_chunks, t_chunks, s_indices, t_indices) in enumerate(zip(
            student_chunks_list, teacher_chunks_list, kept_indices_batch, full_batch_indices
        )):
            t_idx_tensor = torch.tensor(t_indices, device=t_chunks.device)
            s_idx_tensor = torch.tensor(s_indices, device=t_chunks.device)
            
            with torch.no_grad():
                # Teacher global rep uses teacher projection head
                z_raw_teacher = self.teacher_attention_pool(t_chunks, indices=t_idx_tensor)
                # FIX: Use teacher_mu_head instead of mu_head
                z_prime = self.teacher_projector(self.teacher_mu_head(z_raw_teacher)) 
                
                z_teacher_partial = self.teacher_attention_pool(t_chunks[s_idx_tensor], indices=t_idx_tensor[s_idx_tensor])
            
            # Student Branch
            z_partial = self.attention_pool(s_chunks, indices=t_idx_tensor[s_idx_tensor])
            
            # IB bottleneck
            mu = self.mu_head(z_partial)
            log_sigma = self.log_sigma_head(z_partial)
            # FIX: Clamp log_sigma to prevent unbounded variance
            log_sigma = torch.clamp(log_sigma, min=-10, max=2)
            
            if self.training:
                std = torch.exp(log_sigma)
                eps = torch.randn_like(std)
                z_sample = mu + std * eps
            else:
                z_sample = mu

            # Student prediction head for global alignment to prevent collapse
            z_proj = self.projector(z_sample)
            z_global_pred = self.global_predictor(z_proj)

            Z_prime_list.append(z_prime)
            Z_proj_list.append(z_global_pred)
            
            # Apply predictors to student representations for alignment
            Z_inter_s_list.append(self.inter_predictor(z_partial))
            Z_inter_t_list.append(z_teacher_partial)
            
            chunks_s_list.append(self.local_predictor(s_chunks))
            chunks_t_list.append(t_chunks[s_idx_tensor]) 
            
            mu_list.append(mu)
            log_sigma_list.append(log_sigma)

        return (
            torch.stack(Z_prime_list),
            torch.stack(Z_proj_list),
            torch.stack(Z_inter_s_list),
            torch.stack(Z_inter_t_list),
            chunks_s_list,
            chunks_t_list,
            torch.stack(mu_list),
            torch.stack(log_sigma_list),
            self.log_s,
        )


class DocumentClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, tokenizer, attn_pool: AttentionPooling = None,
                 num_labels: int = 10, device: str = "cuda"):
        super().__init__()
        self.encoder     = encoder
        self.tokenizer   = tokenizer
        if attn_pool is not None:
            self.attn_pool = attn_pool
        else:
            self.attn_pool = AttentionPooling(dim=768, max_chunks=512)
            
        self.classifier  = nn.Linear(768, num_labels)
        self.to(device)

    def _encode_chunks(self, paragraphs: list, chunk_size: int = 16) -> torch.Tensor:
        if not paragraphs:
             return torch.zeros(1, 768, device=self.encoder.device)
        
        all_vecs = []
        for i in range(0, len(paragraphs), chunk_size):
            batch = paragraphs[i : i + chunk_size]
            enc = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=256, return_tensors="pt",
            ).to(self.encoder.device)
            out = self.encoder(**enc)
            all_vecs.append(out.last_hidden_state[:, 0, :])
        return torch.cat(all_vecs, dim=0)

    def forward(self, paragraphs_batch: list) -> torch.Tensor:
        doc_vecs = []
        for paras in paragraphs_batch:
            # FIX: Get indices from truncation
            paras, indices = truncate_paragraphs(paras, max_chunks=50)
            chunks = self._encode_chunks(paras)
            idx_tensor = torch.tensor(indices, device=chunks.device)
            doc_vecs.append(self.attn_pool(chunks, indices=idx_tensor))
        
        return self.classifier(torch.stack(doc_vecs))
