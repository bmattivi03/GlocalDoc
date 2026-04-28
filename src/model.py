import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer


class GlocalIBModel(nn.Module):
    """
    Teacher-student GlocalIB model.

    Teacher branch: reads full document, stop-gradient (no_grad on encoder forward).
    Student branch: reads masked document, outputs probabilistic (mu, sigma) via
                    a head on the pooled representation, then projects via MLP.

    Both branches share self.encoder weights. The teacher path never contributes
    gradients — only the student path drives parameter updates. This is the
    stop-gradient mechanism that prevents representational collapse (same as BYOL/SimSiam).
    """

    def __init__(self, hidden_dim=256, proj_dim=512, device="cuda"):
        super().__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.encoder = RobertaModel.from_pretrained("roberta-base")

        # Student probabilistic head
        self.mu_head = nn.Linear(768, hidden_dim)
        self.log_sigma_head = nn.Linear(768, hidden_dim)

        # MLP projector: Z(256) → 512 → 768
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, 768),
        )

        # Learnable compression strength; clamped to min 0.01 to prevent collapse
        self.log_beta = nn.Parameter(torch.tensor(0.0))

        self.hidden_dim = hidden_dim
        self.device = device
        self.to(device)

    def _encode_paragraphs(self, paragraphs, stop_grad=False):
        """
        Encode a list of paragraph strings to a single (768,) document vector.

        Each paragraph is tokenized and encoded independently via RoBERTa (chunk-and-pool):
        paragraph → RoBERTa → CLS token (768-dim) → mean-pool across all paragraphs.

        stop_grad=True: teacher branch — encoder runs under torch.no_grad(),
                        so no gradients flow through this path.
        """
        para_vecs = []
        for para in paragraphs:
            enc = self.tokenizer(
                para,
                max_length=512,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            ).to(self.device)
            if stop_grad:
                with torch.no_grad():
                    out = self.encoder(**enc)
            else:
                out = self.encoder(**enc)
            cls_vec = out.last_hidden_state[:, 0, :]  # (1, 768)
            para_vecs.append(cls_vec)
        return torch.stack(para_vecs).squeeze(1).mean(0)  # (768,)

    def forward(self, full_paragraphs_batch, masked_paragraphs_batch):
        """
        Args:
            full_paragraphs_batch:   list[list[str]] — teacher input (all paragraphs)
            masked_paragraphs_batch: list[list[str]] — student input (20-40% dropped)

        Returns:
            Z_prime (B, 768) — teacher document representations (stop-grad)
            mu      (B, 256) — student distribution means
            sigma   (B, 256) — student distribution std devs
            Z_proj  (B, 768) — student projections (aligned to Z_prime)
            beta    scalar   — current compression strength (clamped >= 0.01)
        """
        batch_z_prime, batch_mu, batch_sigma, batch_z_proj = [], [], [], []

        for full_paras, masked_paras in zip(full_paragraphs_batch, masked_paragraphs_batch):
            # Teacher branch — stop-gradient, full document
            z_prime = self._encode_paragraphs(full_paras, stop_grad=True)   # (768,)

            # Student branch — trainable, masked document
            h = self._encode_paragraphs(masked_paras, stop_grad=False)      # (768,)
            mu = self.mu_head(h)                                             # (256,)
            sigma = torch.exp(self.log_sigma_head(h))                        # (256,)

            # Reparameterization trick: z = mu + sigma * eps
            z = mu + sigma * torch.randn_like(mu)
            z_proj = self.projector(z)                                       # (768,)

            batch_z_prime.append(z_prime)
            batch_mu.append(mu)
            batch_sigma.append(sigma)
            batch_z_proj.append(z_proj)

        Z_prime = torch.stack(batch_z_prime)   # (B, 768)
        mu      = torch.stack(batch_mu)        # (B, 256)
        sigma   = torch.stack(batch_sigma)     # (B, 256)
        Z_proj  = torch.stack(batch_z_proj)    # (B, 768)
        beta    = torch.clamp(torch.exp(self.log_beta), min=0.01)

        return Z_prime, mu, sigma, Z_proj, beta


class DocumentClassifier(nn.Module):
    """
    Fine-tuning classifier for multi-label ECtHR classification.

    Full fine-tuning: all encoder weights + linear head are trainable.
    Uses the same chunk-and-pool strategy as GlocalIBModel for consistency —
    each paragraph encoded independently, then mean-pooled into a document vector.

    Initialized from a pre-trained encoder (GlocalIB student or MLM baseline).
    """

    def __init__(self, encoder, tokenizer, num_labels=10, device="cuda"):
        super().__init__()
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.classifier = nn.Linear(768, num_labels)
        self.device = device
        self.to(device)

    def _encode_paragraphs(self, paragraphs):
        para_vecs = []
        for para in paragraphs:
            enc = self.tokenizer(
                para,
                max_length=512,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            ).to(self.device)
            out = self.encoder(**enc)
            cls_vec = out.last_hidden_state[:, 0, :]  # (1, 768)
            para_vecs.append(cls_vec)
        return torch.stack(para_vecs).squeeze(1).mean(0)  # (768,)

    def forward(self, paragraphs_batch):
        """
        Args:
            paragraphs_batch: list[list[str]]

        Returns:
            (B, num_labels) sigmoid probabilities
        """
        doc_vecs = torch.stack([
            self._encode_paragraphs(paras) for paras in paragraphs_batch
        ])  # (B, 768)
        return torch.sigmoid(self.classifier(doc_vecs))  # (B, 10)
