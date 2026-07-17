"""Survival and trajectory metrics."""
import numpy as np
from scipy.stats import spearmanr

def concordance_index(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    comparable = 0
    concordant = 0.0
    for i in range(len(time)):
        for j in range(len(time)):
            if event[i] and time[i] < time[j]:
                comparable += 1
                concordant += float(risk[i] > risk[j]) + 0.5 * float(risk[i] == risk[j])
    return concordant / comparable if comparable else 0.5

def spearman(values: np.ndarray, reference: np.ndarray) -> float:
    return float(spearmanr(values, reference).statistic)
