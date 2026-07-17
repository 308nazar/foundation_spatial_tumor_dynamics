# Foundation Spatial Tumor Dynamics

This package provides a multimodal latent-bottleneck model for spatial transcriptomics, whole-slide histopathology, CT, single-cell RNA-seq and bulk genomics. The implementation includes resolution-aware cross-attention, modality dropout, pseudotime utilities, downstream heads, clinical metrics and atlas-oriented data records.

## Installation

Use Python 3.11 with `pip install -r requirements.txt`, or create the supplied conda environment. The Docker image uses CUDA 12.1 and PyTorch 2.3.1.

## Data

Canonical cohort links are listed in `datasets.txt`. Downloaded data should be converted to patient-level records with modality tensors and coordinate arrays. The paper does not provide release hashes; manifests should be hashed locally before training.

## Training

The selected pretraining configuration is in `configs/experiment/main.yaml`: 16 A100 80GB GPUs, batch 64, 200,000 steps, AdamW learning rate 1e-4, cosine decay and 5,000 warmup steps. Run `PYTHONPATH=src python -m foundation_spatial_tumor_dynamics.cli.train --steps 2` for a local functional check; full training uses the values in the main configuration.

## Evaluation

The metric modules expose AUROC, balanced accuracy, Dice-compatible array inputs, Pearson/Spearman association, concordance, bootstrap intervals and clinical reclassification measures. Expected headline values are AUC 0.832 ± 0.027, balanced accuracy 0.927 ± 0.017, Dice 0.835 ± 0.029, survival C-index 0.722 ± 0.034 and pseudotime/PFS Spearman 0.467 ± 0.057 as reported in the manuscript.

## Compute

Pretraining is reported at 2,340 A100-GPU-hours on 16 A100 80GB GPUs over approximately six days. Single-patient inference is reported at 2.5 seconds on one A100 with five modalities.

## License

MIT.
