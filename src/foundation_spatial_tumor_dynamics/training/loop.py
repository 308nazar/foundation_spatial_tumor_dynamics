"""Modality-dropout pretraining loop from Algorithm 2."""
from collections.abc import Iterator
import torch
from torch import nn
from ..data.types import MODALITIES, PatientRecord

def masked_streams(patient: PatientRecord, probability: float, generator: torch.Generator | None = None) -> dict[str, torch.Tensor]:
    streams = dict(patient.tokens)
    if len(streams) > 1 and torch.rand((), generator=generator).item() < probability:
        selected = MODALITIES[torch.randint(len(MODALITIES), (), generator=generator).item()]
        streams.pop(selected, None)
    if not streams:
        streams = dict(patient.tokens)
    return streams

def train_steps(model: nn.Module, records: Iterator[PatientRecord], optimizer: torch.optim.Optimizer, steps: int, dropout: float = 0.2) -> list[float]:
    model.train()
    losses: list[float] = []
    iterator = iter(records)
    for _ in range(steps):
        patient = next(iterator)
        streams = {k: v.unsqueeze(0) for k, v in masked_streams(patient, dropout).items()}
        outputs = model(streams)
        loss = outputs["response"].pow(2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses
