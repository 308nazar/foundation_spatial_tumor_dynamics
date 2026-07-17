import numpy as np
import torch
from foundation_spatial_tumor_dynamics.data.synthetic import make_patient
from foundation_spatial_tumor_dynamics.models.foundation import FoundationModel
from foundation_spatial_tumor_dynamics.metrics.classification import auc
from foundation_spatial_tumor_dynamics.metrics.survival import spearman

def test_patient_and_forward() -> None:
    patient = make_patient(0, hidden=8, tokens=3)
    model = FoundationModel(hidden_dim=8, latent_count=4, layers=1, heads=2)
    streams = {k: v.unsqueeze(0) for k, v in patient.tokens.items()}
    result = model(streams)
    assert result["latent"].shape == (1, 4, 8)

def test_metrics() -> None:
    assert auc(np.array([0, 1]), np.array([0.1, 0.9])) == 1.0
    assert spearman(np.arange(4), np.arange(4)) > 0.99

def test_training_smoke() -> None:
    model = FoundationModel(hidden_dim=8, latent_count=4, layers=1, heads=2)
    patient = make_patient(0, hidden=8, tokens=3)
    output = model({k: v.unsqueeze(0) for k, v in patient.tokens.items()})
    loss = output["response"].pow(2).mean()
    assert torch.isfinite(loss)
