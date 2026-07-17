"""Resolution-aware coordinate embeddings from Methods §5.2, Eq. (5)."""
import torch
from torch import nn

class ResolutionPositionEmbedding(nn.Module):
    def __init__(self, hidden_dim: int, resolutions: int = 5) -> None:
        super().__init__()
        self.resolution = nn.Embedding(resolutions, hidden_dim)
        self.coordinate = nn.Sequential(nn.Linear(3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, coordinates: torch.Tensor, resolution: torch.Tensor) -> torch.Tensor:
        return self.coordinate(coordinates) + self.resolution(resolution)
