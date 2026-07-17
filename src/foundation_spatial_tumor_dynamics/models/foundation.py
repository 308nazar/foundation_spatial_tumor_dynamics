"""Five-stream latent bottleneck foundation model."""
from collections.abc import Mapping
import torch
from torch import nn
from .cross_attention import CrossAttentionBlock
from .position import ResolutionPositionEmbedding

class FoundationModel(nn.Module):
    def __init__(self, hidden_dim: int = 1024, latent_count: int = 512, layers: int = 12, heads: int = 16, modalities: tuple[str, ...] = ("st", "wsi", "ct", "scrna", "geno")) -> None:
        super().__init__()
        self.modalities = modalities
        self.projections = nn.ModuleDict({m: nn.LazyLinear(hidden_dim) for m in modalities})
        self.position = ResolutionPositionEmbedding(hidden_dim)
        self.latents = nn.Parameter(torch.randn(1, latent_count, hidden_dim) * 0.02)
        self.blocks = nn.ModuleList(CrossAttentionBlock(hidden_dim, heads, 0.1) for _ in range(layers))
        self.norm = nn.LayerNorm(hidden_dim)
        self.response = nn.Linear(hidden_dim, 1)
        self.stage = nn.Linear(hidden_dim, 4)

    def encode(self, streams: Mapping[str, torch.Tensor], positions: Mapping[str, torch.Tensor] | None = None) -> tuple[torch.Tensor, list[torch.Tensor]]:
        tokens: list[torch.Tensor] = []
        for index, modality in enumerate(self.modalities):
            if modality not in streams:
                continue
            value = self.projections[modality](streams[modality])
            if positions and modality in positions:
                resolution = torch.full((value.shape[0],), index, dtype=torch.long, device=value.device)
                value = value + self.position(positions[modality], resolution)
            tokens.append(value)
        if not tokens:
            raise ValueError("At least one modality is required")
        bank = torch.cat(tokens, dim=1)
        latent = self.latents.expand(bank.shape[0], -1, -1)
        maps: list[torch.Tensor] = []
        for block in self.blocks:
            latent, weights = block(latent, bank)
            maps.append(weights)
        return self.norm(latent), maps

    def forward(self, streams: Mapping[str, torch.Tensor], positions: Mapping[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        latent, maps = self.encode(streams, positions)
        pooled = latent.mean(dim=1)
        return {"latent": latent, "response": self.response(pooled), "stage": self.stage(pooled), "attention": maps}
