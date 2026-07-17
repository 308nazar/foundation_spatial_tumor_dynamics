"""Perceiver cross-attention stack from Methods §5.2 and Algorithm 1."""
import torch
from torch import nn

class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_k = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 4, hidden_dim))

    def forward(self, latent: torch.Tensor, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.norm_q(latent)
        key = self.norm_k(tokens)
        update, weights = self.attn(query, key, key, need_weights=True)
        result = latent + update
        return result + self.ff(result), weights
