"""Objective terms from Methods §5.3, Eq. (6)."""
import torch
from torch import nn
import torch.nn.functional as F

class MaskedReconstruction(nn.Module):
    def forward(self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        error = (prediction - target).pow(2)
        if mask is not None:
            error = error.masked_select(mask)
        return error.mean()

class PairContrastive(nn.Module):
    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        a = F.normalize(left, dim=-1)
        b = F.normalize(right, dim=-1)
        logits = a @ b.T / self.temperature
        target = torch.arange(logits.shape[0], device=logits.device)
        return (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2

class CombinedObjective(nn.Module):
    def __init__(self, reconstruction: float = 1.0, contrastive: float = 0.5, downstream: float = 0.2) -> None:
        super().__init__()
        self.weights = (reconstruction, contrastive, downstream)
        self.reconstruction = MaskedReconstruction()
        self.contrastive = PairContrastive()

    def forward(self, recon: torch.Tensor, target: torch.Tensor, left: torch.Tensor, right: torch.Tensor, prediction: torch.Tensor, label: torch.Tensor) -> dict[str, torch.Tensor]:
        terms = {"reconstruction": self.reconstruction(recon, target), "contrastive": self.contrastive(left, right), "downstream": F.binary_cross_entropy_with_logits(prediction.squeeze(-1), label.float())}
        terms["total"] = self.weights[0] * terms["reconstruction"] + self.weights[1] * terms["contrastive"] + self.weights[2] * terms["downstream"]
        return terms
