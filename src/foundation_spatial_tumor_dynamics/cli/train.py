"""Training command line entry point."""
import argparse
import torch
from ..data.synthetic import make_patient
from ..models.foundation import FoundationModel
from ..training.loop import train_steps
from ..utils.seed import set_seed

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    set_seed(args.seed)
    model = FoundationModel(hidden_dim=64, latent_count=16, layers=2, heads=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    records = (make_patient(i, hidden=64, tokens=8) for i in range(max(args.steps, 2) + 1))
    values = train_steps(model, records, optimizer, args.steps)
    print({"steps": len(values), "loss": values[-1]})

if __name__ == "__main__":
    main()
