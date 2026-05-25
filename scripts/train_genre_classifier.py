#!/usr/bin/env python
# antreneaza clasificatorul xgboost de gen muzical pe datasetul kaggle si il salveaza
import os
import pickle
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT     = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data")
TRAIN_PATH = os.path.join(DATA_DIR, "training_dataset.parquet")
MODEL_PATH = os.path.join(DATA_DIR, "genre_classifier.pkl")

sys.path.insert(0, os.path.join(ROOT, "src"))
from ai_based_music_recommendation.tools.explainability_tools import _consolidate_genre

NUMERIC_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
]
# mode si key adauga semnal de gen dincolo de cele 9 caracteristici audio
CLASSIFIER_FEATURES = NUMERIC_FEATURES + ["mode", "key"]
MIN_SAMPLES = 100
CV_FOLDS    = 5

# etichete care nu sunt genuri reale si trebuie excluse din antrenament
DROP_TAGS = {
    "children's music", "children", "kids", "disney", "show-tunes",
    "happy", "sad", "romance", "party", "guitar", "british",
    "female_vocalists", "boy band", "adult standards", "a capella",
    "80s", "today", "experimental", "brazil", "chillout",
    "industrial", "goth", "gothic",
    "gospel", "afrobeats", "punk",
}


# incarca datele, antreneaza modelul cu validare incrucisata si il salveaza
def main() -> None:
    if not os.path.exists(TRAIN_PATH):
        print(f"training_dataset.parquet not found at {TRAIN_PATH}")
        print("Run scripts/merge_kaggle.py first.")
        sys.exit(1)

    print("Loading training_dataset.parquet ...")
    df = pd.read_parquet(TRAIN_PATH)
    print(f"  {len(df):,} total songs")

    # consolideaza genurile la categorii largi si elimina inregistrarile fara gen sau cu nan
    df["genre_consolidated"] = df["track_genre"].apply(_consolidate_genre)
    df = df[df["genre_consolidated"] != "unknown"].dropna(subset=NUMERIC_FEATURES)
    print(f"  {len(df):,} songs after removing unknowns / NaN features")

    # elimina etichetele care nu sunt genuri muzicale propriu-zise
    before_tag_drop = len(df)
    normalised = df["genre_consolidated"].str.lower().str.replace("'", "'", regex=False)
    df = df[~normalised.isin(DROP_TAGS)].reset_index(drop=True)
    print(f"  Dropped {before_tag_drop - len(df):,} rows (non-genre tags)")

    # elimina clasele cu prea putine exemple pentru a fi invatate
    counts = df["genre_consolidated"].value_counts()
    keep   = counts[counts >= MIN_SAMPLES].index
    dropped_classes = counts[counts < MIN_SAMPLES]
    df = df[df["genre_consolidated"].isin(keep)].reset_index(drop=True)

    n_classes = df["genre_consolidated"].nunique()
    print(f"\nDropped {len(dropped_classes)} classes with < {MIN_SAMPLES} songs "
          f"({dropped_classes.sum():,} rows removed)")
    print(f"Training on {n_classes} classes, {len(df):,} songs\n")

    print("Genre distribution (training set):")
    for genre, cnt in df["genre_consolidated"].value_counts().items():
        bar = "█" * min(35, int(cnt / df["genre_consolidated"].value_counts().max() * 35))
        print(f"  {genre:<25} {cnt:>7,}  {bar}")

    # inlocuieste -1 (cheie sau mod necunoscut) cu nan pe care xgboost il gestioneaza nativ
    for col in ["mode", "key"]:
        if col in df.columns:
            df[col] = df[col].replace(-1, np.nan)
        else:
            df[col] = np.nan

    X = df[CLASSIFIER_FEATURES].values.astype(float)
    le = LabelEncoder()
    y = le.fit_transform(df["genre_consolidated"])

    print(f"\n--- {CV_FOLDS}-fold stratified cross-validation ---")
    model_cv = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=42,
        verbosity=0,
    )
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model_cv, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
    print(f"  Accuracy per fold: {[f'{s:.3f}' for s in cv_scores]}")
    print(f"  Mean accuracy:     {cv_scores.mean():.3f}  (+/- {cv_scores.std():.3f})")

    # raport detaliat pe clase pentru ultimul fold de validare
    print("\n--- Per-class report (last fold) ---")
    train_idx, test_idx = list(skf.split(X, y))[-1]
    model_cv.fit(X[train_idx], y[train_idx])
    y_pred = model_cv.predict(X[test_idx])
    print(classification_report(
        y[test_idx], y_pred,
        target_names=le.classes_,
        digits=3,
        zero_division=0,
    ))

    # antreneaza modelul final pe intregul dataset si il salveaza
    print("--- Training final model on full dataset ---")
    final_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=42,
        verbosity=0,
    )
    final_model.fit(X, y, verbose=False)
    print("  Done.")

    with open(MODEL_PATH, "wb") as fh:
        pickle.dump({"model": final_model, "encoder": le}, fh)
    print(f"\nSaved -> {MODEL_PATH}")
    print(f"Classes: {list(le.classes_)}")
    print("\nDone.")


if __name__ == "__main__":
    main()
