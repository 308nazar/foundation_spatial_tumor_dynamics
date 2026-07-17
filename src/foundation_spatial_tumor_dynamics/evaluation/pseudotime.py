"""Cross-sectional manifold inference from Methods §5.4 and Algorithm 3."""
import numpy as np
from sklearn.decomposition import PCA
from ..metrics.survival import spearman

def infer_pseudotime(latents: np.ndarray, stages: np.ndarray) -> np.ndarray:
    reduced = PCA(n_components=min(2, latents.shape[1])).fit_transform(latents)
    low = reduced[stages == stages.min()].mean(axis=0)
    high = reduced[stages == stages.max()].mean(axis=0)
    direction = high - low
    values = (reduced - low) @ direction
    span = float(values.max() - values.min())
    return (values - values.min()) / span if span > 0 else np.zeros(len(values))

def validate_pseudotime(values: np.ndarray, stages: np.ndarray) -> float:
    return spearman(values, stages)
