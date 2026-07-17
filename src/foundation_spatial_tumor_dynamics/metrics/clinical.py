"""Clinical utility metrics from Results §2.8 and Table 11."""
import numpy as np

def reclassification_improvement(base: np.ndarray, model: np.ndarray, outcome: np.ndarray) -> float:
    event = outcome.astype(bool)
    return float((model[event].mean() - base[event].mean()) - (model[~event].mean() - base[~event].mean()))

def integrated_discrimination(base: np.ndarray, model: np.ndarray, outcome: np.ndarray) -> float:
    event = outcome.astype(bool)
    return float((model[event].mean() - model[~event].mean()) - (base[event].mean() - base[~event].mean()))

def decision_net_benefit(score: np.ndarray, outcome: np.ndarray, threshold: float = 0.3) -> float:
    positive = score >= threshold
    n = len(score)
    tp = np.sum(positive & outcome.astype(bool))
    fp = np.sum(positive & ~outcome.astype(bool))
    return float(tp / n - fp / n * threshold / (1 - threshold))

def hosmer_lemeshow(score: np.ndarray, outcome: np.ndarray, bins: int = 10) -> float:
    edges = np.quantile(score, np.linspace(0, 1, bins + 1))
    statistic = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (score >= low) & (score <= high)
        if mask.sum() == 0:
            continue
        expected = score[mask].sum()
        observed = outcome[mask].sum()
        statistic += float((observed - expected) ** 2 / (expected + 1e-8))
    return statistic
