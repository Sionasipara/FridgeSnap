# models/

Trained weights. Gitignored.

## Contents
- `detector/`: fine-tuned YOLO ingredient detector
- `freshness/`: spoilage-risk classifier
- `embeddings/`: recipe embedding index and vectorizer

## Getting weights
Weights are stored in <Git LFS / Hugging Face / UTMIST bucket: fill in>.
Run `bash scripts/download_models.sh` to fetch them.

## Naming
`<component>_<arch>_<YYYYMMDD>_<short-git-hash>.pt`
(e.g. `detector_yolov8s_20261101_a1b2c3d.pt`)

Record training config, dataset version, and metrics in the matching file under
`evaluation/reports/` for every released checkpoint.
