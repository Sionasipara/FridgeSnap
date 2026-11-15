# FridgeSnap 🧊📸

**CV-driven meal recommendation to reduce household food waste.**

FridgeSnap detects ingredients from a photo of your fridge, estimates how close
each item is to spoiling, and recommends recipes ranked not just by ingredient
match but by how well they use up food that is about to go bad.

A [UTMIST](https://utmist.gitlab.io/) project.

## Why

Households throw away food that spoils before it's used, often because they
forget what they have. Existing "what can I make" tools require manual entry
and rank purely by ingredient overlap, with no notion of what is at risk of
being wasted. FridgeSnap removes the manual entry and adds a waste-aware
ranking objective, then measures whether it actually changes behaviour.

## How it works

1. **Detection & Freshness**: a fine-tuned YOLO detector identifies ingredients
   and quantities; a classifier estimates spoilage risk from visual cues
   (discoloration, wilting, mold).
2. **Inventory Construction**: detections and freshness scores are compiled
   into a structured per-user inventory.
3. **Recipe Retrieval**: recipes are embedded from their ingredient lists and
   matched against the inventory via cosine similarity.
4. **Waste-Aware Re-Ranking**: match scores are re-weighted by the urgency of
   the ingredients each recipe uses, boosting recipes that consume near-expiry
   items.

```
fridge photo → detector → freshness → inventory → retrieval → re-ranker → recipes
```

## Repository layout

| Path | Purpose |
|---|---|
| `src/fridgesnap/` | Core ML library (detection, freshness, inventory, retrieval, reranking) |
| `api/` | FastAPI backend |
| `web/` | React + Material UI frontend |
| `data/` | Datasets and splits (gitignored; see `data/README.md`) |
| `models/` | Trained weights (gitignored; see `models/README.md`) |
| `configs/` | YAML configs for training and re-ranking |
| `notebooks/` | EDA and tuning notebooks |
| `scripts/` | Data download, training, and indexing entry points |
| `evaluation/` | Metrics, pilot analysis, and reports |
| `docs/` | Architecture, privacy policy, contributing guide |
| `tests/` | Unit and integration tests |
| `docker/` | Dockerfiles for API, web, and training |

## Getting started

```bash
# 1. Clone
git clone https://github.com/<org>/fridgesnap.git
cd fridgesnap

# 2. Python environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Configure
cp .env.example .env

# 4. Download datasets (see data/README.md for licenses)
bash scripts/download_data.sh

# 5. Run the stack
docker compose up
```

- API: http://localhost:8000 (docs at `/docs`)
- Web app: http://localhost:3000

## Tech stack

Python · PyTorch · torchvision · Ultralytics YOLO · OpenCV ·
sentence-transformers / TF-IDF · FastAPI · React · Material UI · Docker

## Datasets

- [Open Images](https://storage.googleapis.com/openimages/web/index.html) (food subset): detector training
- Fresh and Rotten Produce dataset: freshness classifier
- Food.com Recipes and Interactions: recipe embedding index

## Targets

| Metric | Target |
|---|---|
| Detection mAP | ≥ 0.75 |
| Freshness accuracy | ≥ 85% |
| Top-3 recipes rated "makeable" | ≥ 80% |
| API latency (photo → recommendations) | ≤ 500 ms |
| Pilot sessions | ≥ 300 |
| Self-reported waste | Measurable reduction vs. baseline week |

## Timeline

| Phase | Duration | Focus |
|---|---|---|
| 1 | Month 1 | Detection and freshness models |
| 2 | Month 2 | Recipe retrieval and waste-aware re-ranking |
| 3 | Month 3 | Internal pilot and model refinement |
| 4 | Month 4 | API, web app, evaluation, launch |

## Contributing

See [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md). In short: branch from `dev`
using `feature/<subteam>-<name>`, open a PR, get one review, and make sure CI
passes.

## Privacy

Pilot photos are collected only with explicit opt-in and are anonymized before
storage. See [docs/data_privacy.md](docs/data_privacy.md).

## Team

Built by the UTMIST FridgeSnap team. Project lead: Sunny Shi.

## License

See [LICENSE](LICENSE).

## References

Marín et al., "Recipe1M+: A Dataset for Learning Cross-Modal Embeddings for
Cooking Recipes and Food Images," IEEE TPAMI, 43(1), 187–203, 2021.
