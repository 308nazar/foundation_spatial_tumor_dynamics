"""Clinical classification metrics."""
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

def auc(y_true: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(y_true, score))

def accuracy(y_true: np.ndarray, prediction: np.ndarray) -> float:
    return float(accuracy_score(y_true, prediction))

def balanced_accuracy(y_true: np.ndarray, prediction: np.ndarray) -> float:
    return float(balanced_accuracy_score(y_true, prediction))

def bootstrap_auc(y_true: np.ndarray, score: np.ndarray, repeats: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        indices = rng.integers(0, len(y_true), len(y_true))
        if np.unique(y_true[indices]).size < 2:
            continue
        values.append(auc(y_true[indices], score[indices]))
    return float(np.mean(values)), float(np.std(values))
