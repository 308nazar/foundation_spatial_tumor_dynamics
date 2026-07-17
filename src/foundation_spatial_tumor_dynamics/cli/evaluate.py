"""Evaluation command line entry point."""
import argparse
import numpy as np
from ..evaluation.tasks import evaluate_response

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=32)
    args = parser.parse_args()
    rng = np.random.default_rng(2026)
    labels = rng.integers(0, 2, args.n)
    scores = labels + rng.normal(0, 0.5, args.n)
    print(evaluate_response(labels, scores))

if __name__ == "__main__":
    main()
