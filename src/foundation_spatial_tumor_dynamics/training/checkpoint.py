"""Atomic checkpoint persistence."""
from pathlib import Path
import os
import tempfile
import torch

def save_checkpoint(path: str | Path, state: dict[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(state, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()

def load_checkpoint(path: str | Path, device: str = "cpu") -> dict[str, object]:
    value = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(value, dict):
        raise TypeError("Checkpoint must contain a mapping")
    return value
