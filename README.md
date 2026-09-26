# Mechanism-Specific Localization Repair under Real Acquisition Shift in Chest Radiographs

Public snapshot of the analysis code, result tables and paper figures for the paper
*"Mechanism-Specific Localization Repair under Real Acquisition Shift in Chest Radiographs."*

Weakly supervised CAM localization in chest radiographs fails under real photographic
capture shift through two distinct mechanisms — **readout drift** and **capture geometry** —
that synthetic degradations overestimate by 4–6× and that medical fine-tuning does not
mitigate. Matching repair to mechanism (PAIR-Loc paired distillation; RectNet homography
rectification; fixed per-finding routing) yields +0.15–0.20 IoU@0.45 in the real
re-photography domain with explicitly bounded wrong-repair rates.

## Layout

- `workspace/` — experiment and analysis scripts (Python; paths point at the original GPU workspace)
- `results/` — result tables (CSV) and checksum manifests from the frozen evaluation
- `paper_fig/` — paper figures and the scripts that generate them

## Scope notes

- This is a curated export: internal reports, logs and working notes are **not** included.
- Result CSVs reference patient/study identifiers of the underlying public datasets
  (CheXpert / CheXphoto / NIH ChestX-ray14); no imaging data is redistributed here.
- Generated from the private development repository at snapshot `5095088`.
