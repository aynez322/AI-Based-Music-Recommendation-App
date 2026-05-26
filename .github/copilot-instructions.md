# Copilot instructions

## Commands
- `python -m pip install -e .`
- `run_recommendations "<song title>" "<artist>"` (omit args to get prompts)
- Data pipeline (order):
  - `python scripts/prepare_zenodo.py`
  - `python scripts/merge_kaggle.py`
  - `python scripts/train_genre_classifier.py`
  - `python scripts/build_final_dataset.py`

## Architecture
- `src/ai_based_music_recommendation/main.py` is the CLI entry: it pre-checks the song with `SongLookupTool`, runs the crew, prints the report, and surfaces `output/recommendations.md`.
- `src/ai_based_music_recommendation/crew.py` wires the CrewAI flow (sequential) and loads definitions from `config/agents.yaml` + `config/tasks.yaml`.
- `src/ai_based_music_recommendation/tools/dataset_tools.py` provides lookup, genre fingerprinting, and similarity search over `data/merged_dataset.parquet` and `data/genre_fingerprints`.
- `src/ai_based_music_recommendation/tools/explainability_tools.py` trains/loads the XGBoost classifier (`data/genre_classifier.pkl`) and computes SHAP explanations.

## Conventions
- The 9 audio features in `NUMERIC_FEATURES` are the shared contract across dataset tools, fingerprints, similarity, and reports—update all usages together.
- `data/genre_fingerprints` is parsed as Markdown with `## <genre>` headings and a `key signals:` line; keep that format if regenerating.
- Data/model paths resolve relative to the repo `data/` directory and are cached globally in the tools; avoid mutating cached DataFrames in-place.
