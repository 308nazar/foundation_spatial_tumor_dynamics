"""Typed experiment configuration."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml

@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int = 1024
    latent_count: int = 512
    layers: int = 12
    heads: int = 16
    dropout: float = 0.1
    modalities: tuple[str, ...] = ("st", "wsi", "ct", "scrna", "geno")

@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 64
    steps: int = 200000
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    warmup_steps: int = 5000
    dropout_rate: float = 0.2
    grad_clip: float = 1.0
    seed: int = 2026

@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()

def load_config(path: str | Path) -> ExperimentConfig:
    data: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
    model = ModelConfig(**{k: v for k, v in data.get("model", {}).items() if k in ModelConfig.__annotations__})
    train = TrainConfig(**{k: v for k, v in data.get("train", {}).items() if k in TrainConfig.__annotations__})
    return ExperimentConfig(model=model, train=train)
