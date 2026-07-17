# Project Context

project_name       : foundation_spatial_tumor_dynamics [HIGH]
domain             : computational oncology — multimodal cellular-resolution NSCLC response modelling [HIGH]
framework          : PyTorch 2.x + plain torch.nn, with PyTorch distributed and YAML configuration [MEDIUM]
venue              : Nature Communications [MEDIUM]
primary_datasets   : 15 cohort/reference entries (see §6) [HIGH]
compute_target     : 16 NVIDIA A100 80GB GPUs, 2,340 GPU-hours, approximately six wall-clock days; 2.5 s/patient inference on one A100 [HIGH]
hparams_reference  : Methods §5.3 and §5.6; Supplementary Tables SA-2, SA-4, SD-1 [HIGH]
supp_path          : none found beside the supplied PDF; supplementary tables and sections are embedded in the PDF [HIGH]
extra_signals      : four numbered algorithms; six downstream tasks; modality-dropout p=0.2; public-data-only claim; atlas/API and h5ad export described

NEEDS_USER_DECISION: 0
Ready to proceed to Turn 1? awaiting approval.

## 1. Project name

Derived value: `foundation_spatial_tumor_dynamics`.

Source: title on PDF page 1, “A foundation model integrating spatial transcriptomics, imaging, and genomics for cellular-resolution tumor burden dynamics and treatment response mapping in lung cancer”. Stopwords were removed and four content groups retained. Confidence: HIGH.

## 2. Supplementary material

No sibling supplementary PDF, DOCX, TEX, or archive was present under `paper/`; supplementary sections SA–SD and Tables 6–28 are included in the supplied PDF. Source: PDF pages 62–80 and table headings. Confidence: HIGH.

## 3. Domain

Computational oncology, specifically multimodal representation learning for cellular-resolution NSCLC tumor burden and immune-checkpoint-blockade response. Source: Abstract; Introduction; Results §§2.1–2.9. Confidence: HIGH.

## 4. Framework

PyTorch 2.x with `torch.nn` is selected as the implementation stack. The paper names Perceiver-IO, AdamW, UMAP, and modality-specific neural encoders but does not state a software framework or version. Source: Methods §§5.2–5.6. Confidence: MEDIUM; the default is used because no competing framework is specified.

## 5. Venue

Nature Communications is the best-supported venue inference: the manuscript explicitly discusses the Nature Communications results-first format and the Cancer Cell Atlases cross-journal Collection. Source: Introduction, final paragraph; Data/code statements in Methods §5.9. Confidence: MEDIUM because the title page has a submission placeholder and no confirmed acceptance metadata.

## 6. Primary datasets

The following public cohorts/references are named in Methods §5.1 and Supplementary Table SA-1 (Table 6). Access URLs are canonical landing pages or accession resolvers; exact release versions are not stated and are recorded as `paper-unspecified`.

| Name | Version | License/access | URL |
|---|---|---|---|
| TCGA-LUAD | paper-unspecified | NIH/GDC open | https://portal.gdc.cancer.gov/projects/TCGA-LUAD |
| TCGA-LUSC | paper-unspecified | NIH/GDC open | https://portal.gdc.cancer.gov/projects/TCGA-LUSC |
| CPTAC-LUAD | paper-unspecified | CPTAC/TCIA restricted-open | https://www.cancerimagingarchive.net/collection/cptac-luad/ |
| NSCLC-Radiogenomics | paper-unspecified | TCIA restricted-open | https://www.cancerimagingarchive.net/collection/nsclc-radiogenomics/ |
| LIDC-IDRI | paper-unspecified | TCIA open | https://www.cancerimagingarchive.net/collection/lidc-idri/ |
| National Lung Screening Trial | de-identified release | NCI open | https://cdas.cancer.gov/datasets/nlst/ |
| E-MTAB-13530 | paper-unspecified | CC BY 4.0 | https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-13530 |
| GSE131907 | paper-unspecified | GEO open | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE131907 |
| GSE153935 | paper-unspecified | GEO open | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE153935 |
| GSE189144 | paper-unspecified | GEO open | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE189144 |
| Human Lung Cell Atlas v2 | v2 | CC BY 4.0 | https://cellxgene.cziscience.com/collections/7a0f7f2d-7d8f-4d2d-9f48-5e2f4f3e2f7c |
| Tabula Sapiens | paper-unspecified | CC BY 4.0 | https://tabula-sapiens-portal.ds.czbiohub.org/ |
| cBioPortal Riaz 2017 | 2017 cohort | cBioPortal open | https://www.cbioportal.org/study/summary?id=nsclc_riaz_2017 |
| cBioPortal Hugo 2016 | 2016 cohort | cBioPortal open | https://www.cbioportal.org/study/summary?id=melanoma_hugo_2016 |
| cBioPortal supplemental ICB cohorts | four-study aggregate | cBioPortal open | https://www.cbioportal.org/ |

Counts stated in Table SA-1 include 522 TCGA-LUAD, 504 TCGA-LUSC, 110 CPTAC-LUAD, 211 NSCLC-Radiogenomics, 1,018 LIDC-IDRI, 75,000+ NLST scans, 40 E-MTAB-13530 sections, 208,506 GSE131907 cells, 79,000 GSE189144 cells, 2.4M HLCA cells, 500,000 Tabula Sapiens cells, 65 Riaz PFS patients, 122 Hugo PFS patients, and 2,111 aggregate supplemental ICB patients. Licenses are reproduced from Table SA-1; where the paper says only “open”, the canonical repository terms govern.

## 7. Compute target

Reported target: 16 A100 80GB GPUs, 2,340 A100-GPU-hours, approximately six days for pretraining; downstream fine-tuning is at most 30 A100-GPU-hours per task; pseudo-time fitting is at most one CPU-hour; single-patient five-modality inference is 2.5 seconds on one A100. Source: Results §2.1; Methods §§5.2–5.6; Supplementary Table SA-5. Confidence: HIGH.

## 8. Hyperparameters

Pretraining: Perceiver latent count L=512, hidden width d=1024, depth 12; AdamW; learning rate 1e-4; weight decay 0.05; batch 64 patients; 200,000 steps; cosine decay to zero; 5,000-step linear warmup; gradient norm clip 1.0; modality-dropout p=0.2; loss weights (reconstruction, contrastive, downstream)=(1.0, 0.5, 0.2). Fine-tuning: 2,000 steps; batch 16; AdamW; learning rate 3e-5; weight decay 0.01; 200-step warmup; cosine decay; 90% backbone frozen with final two cross-attention layers and input projections trainable; five restarts and five-fold CV where specified. Source: Methods §§5.2–5.6; Algorithms 1–4; Supplementary Tables SA-2, SA-4, SD-1. Confidence: HIGH.

## 9. Extra signals

The PDF contains four numbered algorithms, Theorems 1–3, six downstream tasks, 25 baseline families, cumulative-modality and FLOP-matched ablations, mask-rate and architecture sweeps, cross-cancer transfer, and scaling studies. It claims public data only and describes release of preprocessing, weights, and a queryable atlas API. It reports no private hospital data and includes a representation/fairness discussion in Supplementary Table 23. Source: Results §§2.1–2.9; Methods §5.9; Supplementary Tables 12–28. Confidence: HIGH.
