# data/

Local datasets and manifests. **Nothing in here is committed to Git** except
this file and `.gitkeep` placeholders.

## Layout
- `raw/`: untouched downloads
  - `open_images_food/`: detection images + bounding boxes (CC-BY 2.0/4.0)
  - `fresh_rotten_produce/`: freshness-labeled produce images
  - `food_com_recipes/`: recipes with ingredient lists (CSV)
  - `fridge_photos_internal/`: small real-world validation set
- `processed/`: cleaned, resized, or converted data
- `splits/`: train/val/test manifests (CSV/JSON) so everyone uses the same splits
- `pilot/`: anonymized pilot data. **Opt-in only; never publish.**

## Getting the data
Run `bash scripts/download_data.sh`. Check each dataset's license before use or
redistribution.

## Rules
- Never commit raw images or CSVs.
- Never commit anything containing user photos or identifiers.
- Always generate splits with the provided script and a fixed seed.
