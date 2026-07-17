"""Six downstream task adapters."""
from dataclasses import dataclass
import numpy as np
import torch
from ..metrics.classification import auc, balanced_accuracy
from ..metrics.survival import concordance_index

@dataclass(frozen=True)
class TaskResult:
    name: str
    value: float
    spread: float

def evaluate_response(labels: np.ndarray, scores: np.ndarray) -> TaskResult:
    return TaskResult("icb_response_auc", auc(labels, scores), 0.0)

def evaluate_subtype(labels: np.ndarray, logits: np.ndarray) -> TaskResult:
    prediction = logits.argmax(axis=1)
    return TaskResult("subtype_balanced_accuracy", balanced_accuracy(labels, prediction), 0.0)

def evaluate_survival(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> TaskResult:
    return TaskResult("overall_survival_c_index", concordance_index(time, event, risk), 0.0)

def evaluate_segmentation(target: torch.Tensor, prediction: torch.Tensor) -> TaskResult:
    overlap = (target.bool() & prediction.bool()).sum().float()
    total = target.bool().sum() + prediction.bool().sum()
    score = float((2 * overlap / total.clamp_min(1)).item())
    return TaskResult("ct_dice", score, 0.0)

def evaluate_expression(target: np.ndarray, prediction: np.ndarray) -> TaskResult:
    correlation = np.corrcoef(target.reshape(-1), prediction.reshape(-1))[0, 1]
    return TaskResult("st_expression_pearson", float(correlation), 0.0)
