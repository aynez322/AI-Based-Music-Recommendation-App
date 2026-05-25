#!/usr/bin/env python
# combina datasetul zenodo cu cel kaggle si salveaza fisierul parquet final
import os
import pickle
import sys

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT     = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data")

TRAIN_PATH  = os.path.join(DATA_DIR, "training_dataset.parquet")
ZENODO_PATH = os.path.join(DATA_DIR, "merged_dataset.parquet")
MODEL_PATH  = os.path.join(DATA_DIR, "genre_classifier.pkl")
OUTPUT_PATH = os.path.join(DATA_DIR, "merged_dataset.parquet")

sys.path.insert(0, os.path.join(ROOT, "src"))
from ai_based_music_recommendation.tools.explainability_tools import (
    CLASSIFIER_FEATURES,
    _consolidate_genre,
)
from ai_based_music_recommendation.tools.dataset_tools import NUMERIC_FEATURES


# incarca modelul xgboost si encoder-ul din fisierul pkl
def _load_model():
    with open(MODEL_PATH, "rb") as fh:
        saved = pickle.load(fh)
    return saved["model"], saved["encoder"]


# asigura ca coloanele key si mode exista si au tipul numeric corect
def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["key", "mode"]:
        if col not in df.columns:
            df[col] = -1
    df["key"]  = pd.to_numeric(df["key"],  errors="coerce").fillna(-1).astype(int)
    df["mode"] = pd.to_numeric(df["mode"], errors="coerce").fillna(-1).astype(int)
    return df


# ruleaza clasificatorul xgboost si returneaza genurile prezise ca serie de stringuri
def _predict_genres(df: pd.DataFrame, model, le) -> pd.Series:
    X = np.column_stack([
        df[f].values if f in df.columns else np.full(len(df), np.nan)
        for f in CLASSIFIER_FEATURES
    ]).astype(float)
    X = np.where(X == -1, np.nan, X)
    preds = model.predict(X)
    return pd.Series(le.inverse_transform(preds), index=df.index)


# incarca, combina si salveaza datasetul final din zenodo si kaggle
def main() -> None:
    if not os.path.exists(MODEL_PATH):
        print(f"genre_classifier.pkl not found — run train_genre_classifier.py first.")
        sys.exit(1)
    print("Loading genre classifier ...")
    model, le = _load_model()
    trained_classes = set(le.classes_)
    print(f"  {len(trained_classes)} classes: {sorted(trained_classes)}")

    if not os.path.exists(TRAIN_PATH):
        print(f"training_dataset.parquet not found — run merge_kaggle.py first.")
        sys.exit(1)
    print(f"\nLoading training_dataset.parquet ...")
    df_kaggle = pd.read_parquet(TRAIN_PATH)
    df_kaggle = _standardize(df_kaggle)
    print(f"  {len(df_kaggle):,} total songs")

    df_kaggle["track_genre"] = df_kaggle["track_genre"].apply(_consolidate_genre)
    df_kaggle = df_kaggle[df_kaggle["track_genre"].isin(trained_classes)]
    df_kaggle = df_kaggle.dropna(subset=NUMERIC_FEATURES).reset_index(drop=True)
    print(f"  {len(df_kaggle):,} songs after consolidation to 17 classes")

    if not os.path.exists(ZENODO_PATH):
        print(f"merged_dataset.parquet not found — run prepare_zenodo.py first.")
        sys.exit(1)
    print(f"\nLoading merged_dataset.parquet (zenodo) ...")
    df_zenodo = pd.read_parquet(ZENODO_PATH)
    df_zenodo = _standardize(df_zenodo)
    print(f"  {len(df_zenodo):,} total songs")

    df_zenodo = df_zenodo.dropna(subset=NUMERIC_FEATURES).reset_index(drop=True)
    df_zenodo["track_genre"] = df_zenodo["track_genre"].apply(_consolidate_genre)
    needs_pred = ~df_zenodo["track_genre"].isin(trained_classes)
    print(f"  {(~needs_pred).sum():,} songs keep original consolidated genre")
    print(f"  {needs_pred.sum():,} songs will get XGBoost prediction")

    if needs_pred.any():
        df_zenodo.loc[needs_pred, "track_genre"] = _predict_genres(
            df_zenodo[needs_pred], model, le
        )

    print(f"\nMerging {len(df_zenodo):,} zenodo + {len(df_kaggle):,} Kaggle songs ...")
    keep_cols = ["track_name", "artists", "track_genre", "popularity", "key", "mode"] + NUMERIC_FEATURES
    df_zenodo = df_zenodo[[c for c in keep_cols if c in df_zenodo.columns]]
    df_kaggle = df_kaggle[[c for c in keep_cols if c in df_kaggle.columns]]

    # zenodo este primar, deci randul sau castiga la deduplicare
    combined = pd.concat([df_zenodo, df_kaggle], ignore_index=True)
    print(f"  Before dedup: {len(combined):,}")

    combined["_key"] = (
        combined["track_name"].str.strip().str.lower()
        + "|||"
        + combined["artists"].str.strip().str.lower()
    )
    combined["_has_genre"] = (combined["track_genre"] != "unknown").astype(int)
    combined = (
        combined
        .sort_values(["_has_genre", "popularity"], ascending=[False, False])
        .drop_duplicates(subset="_key", keep="first")
        .drop(columns=["_key", "_has_genre"])
        .reset_index(drop=True)
    )
    print(f"  After dedup:  {len(combined):,}")

    print(f"\nGenre distribution:")
    counts = combined["track_genre"].value_counts()
    for genre, cnt in counts.items():
        bar = "█" * min(35, int(cnt / counts.max() * 35))
        print(f"  {genre:<20} {cnt:>8,}  {bar}")

    combined.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved → {OUTPUT_PATH}")
    print(f"Columns: {list(combined.columns)}")
    print(f"Total songs: {len(combined):,}")
    print("\nDone.")


if __name__ == "__main__":
    main()
