# Implementation Map

| Paper section | Equation/figure/table | File path | Module | Notes |
|---|---|---|---|---|
| Problem formulation | Eq. (1) | src/foundation_spatial_tumor_dynamics/models/problem.py | models | Five-modality input space and task objective |
| Dropout constraint | Eq. (2) | src/foundation_spatial_tumor_dynamics/losses/dropout_bound.py | losses | Expected subset loss and slack metric |
| Generalisation bound | Eq. (3), Theorem 2 | src/foundation_spatial_tumor_dynamics/metrics/generalisation.py | metrics | Empirical bound terms |
| Pseudotime bound | Eq. (4), Theorem 3 | src/foundation_spatial_tumor_dynamics/metrics/pseudotime.py | metrics | Concordance lower-bound calculation |
| Cross-attention | Eq. (5), Theorem 1 | src/foundation_spatial_tumor_dynamics/models/cross_attention.py | models | Resolution-aware Perceiver layer |
| Combined objective | Eq. (6) | src/foundation_spatial_tumor_dynamics/losses/combined.py | losses | Reconstruction, InfoNCE, downstream terms |
| Architecture | Algorithm 1; Fig. 1 | src/foundation_spatial_tumor_dynamics/models/foundation.py | models | L=512, d=1024, 12 layers |
| Modality dropout | Algorithm 2; Fig. 2 | src/foundation_spatial_tumor_dynamics/training/dropout.py | training | Bernoulli p=0.2 and subset-safe aggregation |
| Pseudotemporal manifold | Algorithm 3; Table 2 | src/foundation_spatial_tumor_dynamics/evaluation/pseudotime.py | evaluation | UMAP and anchored principal curve |
| End-to-end workflow | Algorithm 4 | src/foundation_spatial_tumor_dynamics/training/workflow.py | training | Pretraining, fitting, fine-tuning orchestration |
| Main benchmark | Table 1; Fig. 3 | src/foundation_spatial_tumor_dynamics/evaluation/tasks.py | evaluation | Six tasks and baseline comparison |
| Pseudotemporal validation | Table 2 | src/foundation_spatial_tumor_dynamics/metrics/pseudotime.py | metrics | Stage and PFS Spearman rho |
| Architectural ablation | Table 3 | src/foundation_spatial_tumor_dynamics/evaluation/ablations.py | evaluation | Fusion, position embedding, latent and parameter controls |
| Holdout robustness | Table 4 | src/foundation_spatial_tumor_dynamics/evaluation/robustness.py | evaluation | Ten pairwise and five single holdouts |
| External validation | Table 5; Fig. SF-3 | src/foundation_spatial_tumor_dynamics/evaluation/external.py | evaluation | Level A–D, LOCO, heterogeneity |
| Cohort metadata | Table 6 / SA-1 | data/catalog.py; datasets.txt | data | Cohort, modality, license, accession |
| Pretraining sweep | Table 7 / SA-2 | configs/experiment/pretrain_sweep.yaml | configs | Learning rate, batch, mask, decay axes |
| Curves | Table 8 / SA-3 | evaluation/curves.py | evaluation | Loss and validation trajectories |
| Augmentation study | Table 9 / SA-4 | data/augmentations.py | data | CT, WSI, ST, scRNA and genomics policies |
| Compute footprint | Table 10 / SA-5 | evaluation/compute.py | evaluation | GPU-hours, latency, memory |
| Clinical practice | Table 11 | src/foundation_spatial_tumor_dynamics/metrics/clinical.py | metrics | NRI, IDI, decision curve, calibration |
| Cross-cancer transfer | Table 12 | src/foundation_spatial_tumor_dynamics/evaluation/transfer.py | evaluation | Frozen LUAD backbone across TCGA cohorts |
| Parameter scaling | Table 13 | configs/experiment/scaling.yaml; evaluation/scaling.py | configs/evaluation | Five model sizes and FLOP estimates |
| Mask sweep | Table 14 / SB-1 | configs/experiment/mask_sweep.yaml | configs | p in {0,.1,.2,.3,.4} |
| Hyperparameter grid | Table 15 / SB-2 | configs/experiment/hparam_grid.yaml | configs | lr × batch × attention dropout |
| Depth-width grid | Table 16 / SB-3 | configs/experiment/architecture_grid.yaml | configs | Latents, width and layers |
| Encoder swap | Table 17 / SB-4 | configs/experiment/encoder_swap.yaml | configs | CONCH/Phikon, scGPT/Geneformer, CTViT/Sybil |
| Loss sweep | Table 18 / SB-5 | configs/experiment/loss_grid.yaml | configs | Three loss coefficients |
| Per-run results | Table 19 / SC-1 | evaluation/reproducibility.py | evaluation | Five seeds and folds |
| Pairwise holdouts | Table 20 / SC-2 | evaluation/robustness.py | evaluation | All ten two-modality removals |
| Cytokine recovery | Table 21 / SC-3 | metrics/biology.py | metrics | T-tests and FDR correction |
| Forest data | Table 22 / SC-4 | evaluation/meta_analysis.py | evaluation | Random-effects pooled estimates |
| Subgroup analysis | Table 23 / SC-5 | evaluation/subgroups.py | evaluation | Interaction p-values |
| Full hyperparameters | Table 24 / SD-1 | configs/experiment/main.yaml | configs | Selected values and search ranges |
| Scaling laws | Table 25 / SD-2 | evaluation/scaling.py | evaluation | Data, params and compute scaling |
| Reproducibility | Table 26 / SD-3 | training/seeding.py | training | Seeds, variance and confidence intervals |
| Data splits | Table 27 / SD-4 | data/splits.py | data | Patient-level leakage control |
| Clinician benchmarks | Table 28 / SD-5 | metrics/clinical.py | metrics | Published comparator values |
| Attention mechanism | Fig. 4; SF-1–SF-4 | evaluation/attention_maps.py | evaluation | Contact-zone probing and synthetic recovery |
| Atlas export/API | Results §2.9 | src/foundation_spatial_tumor_dynamics/cli/atlas.py | cli | h5ad export and query endpoints |
