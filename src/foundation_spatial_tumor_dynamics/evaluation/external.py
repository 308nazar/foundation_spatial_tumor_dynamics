"""Database-level validation and leave-one-cohort-out aggregation."""
from dataclasses import dataclass
import numpy as np
from ..metrics.classification import auc

@dataclass(frozen=True)
class CohortScore:
    cohort: str
    auc_value: float
    n: int

def score_cohort(name: str, labels: np.ndarray, predictions: np.ndarray) -> CohortScore:
    return CohortScore(name, auc(labels, predictions), len(labels))

def pooled(scores: list[CohortScore]) -> float:
    weights = np.asarray([item.n for item in scores], dtype=float)
    values = np.asarray([item.auc_value for item in scores], dtype=float)
    return float(np.average(values, weights=weights))

def leave_one_out(scores: list[CohortScore]) -> dict[str, float]:
    return {item.cohort: pooled([other for other in scores if other.cohort != item.cohort]) for item in scores}
