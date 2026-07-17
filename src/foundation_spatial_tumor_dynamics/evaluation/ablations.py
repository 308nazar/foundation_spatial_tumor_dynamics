"""Cumulative modality and FLOP-matched ablation runners."""
from dataclasses import dataclass
from collections.abc import Callable, Mapping
import torch

@dataclass(frozen=True)
class AblationPoint:
    name: str
    modalities: tuple[str, ...]
    score: float

def modality_ablation(model: Callable[..., Mapping[str, torch.Tensor]], streams: Mapping[str, torch.Tensor], order: tuple[str, ...]) -> list[AblationPoint]:
    points: list[AblationPoint] = []
    for count in range(1, len(order) + 1):
        selected = {name: streams[name] for name in order[:count] if name in streams}
        output = model(selected)
        score = float(torch.sigmoid(output["response"]).mean().item())
        points.append(AblationPoint(f"{count}_modalities", tuple(selected), score))
    return points
