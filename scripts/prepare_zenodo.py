#!/usr/bin/env python
# converteste fisierul zenodo.csv in formatul parquet folosit de sistemul de recomandare
import ast
import os
import sys

import pandas as pd

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
ZENODO_PATH = os.path.join(DATA_DIR, "datasets", "zenodo.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "merged_dataset.parquet")

NUMERIC_FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]


# extrage primul gen dintr-o lista stringificata de genuri
def _parse_first_genre(raw) -> str:
    if not isinstance(raw, str):
        return "unknown"
    raw = raw.strip()
    if raw in ("[]", "nan", ""):
        return "unknown"
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list) and parsed:
            return str(parsed[0]).strip()
    except (ValueError, SyntaxError):
        pass
    cleaned = raw.strip("[]").replace("'", "").replace('"', "").split(",")
    first = cleaned[0].strip()
    return first if first else "unknown"


# incarca zenodo.csv, curata datele si le salveaza ca parquet
def prepare() -> None:
    if not os.path.exists(ZENODO_PATH):
        print(f"zenodo.csv not found at {ZENODO_PATH}")
        sys.exit(1)

    print(f"Loading {ZENODO_PATH} ...")
    df = pd.read_csv(ZENODO_PATH, low_memory=False)
    print(f"  {len(df):,} rows, {len(df.columns)} columns")

    df = df.rename(columns={
        "name": "track_name",
        "track_artists": "artists",
        "genres": "_raw_genres",
    })

    df["track_genre"] = df["_raw_genres"].apply(_parse_first_genre)

    for col in ["key", "mode"]:
        if col not in df.columns:
            df[col] = -1

    keep = ["track_name", "artists", "track_genre", "popularity", "key", "mode"] + NUMERIC_FEATURES
    missing = [c for c in ["track_name", "artists"] + NUMERIC_FEATURES if c not in df.columns]
    if missing:
        print(f"ERROR: missing columns in zenodo.csv: {missing}")
        sys.exit(1)

    df = df[keep].copy()
    df["key"]  = pd.to_numeric(df["key"],  errors="coerce").fillna(-1).astype(int)
    df["mode"] = pd.to_numeric(df["mode"], errors="coerce").fillna(-1).astype(int)
    df["track_name"] = df["track_name"].fillna("").astype(str)
    df["artists"] = df["artists"].fillna("").astype(str)
    df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0).astype(int)

    before = len(df)
    df = df.dropna(subset=NUMERIC_FEATURES).reset_index(drop=True)
    if before - len(df):
        print(f"  Dropped {before - len(df):,} rows with missing numeric features")

    # deduplicare dupa (track_name, artists), se pastreaza randul cu cel mai mare popularity
    df["_key"] = (
        df["track_name"].str.strip().str.lower()
        + "|||"
        + df["artists"].str.strip().str.lower()
    )
    df["_has_genre"] = (df["track_genre"] != "unknown").astype(int)
    df = df.sort_values(["_has_genre", "popularity"], ascending=[False, False])
    df = df.drop_duplicates(subset="_key", keep="first")
    df = df.drop(columns=["_key", "_has_genre"]).reset_index(drop=True)

    print(f"  After dedup: {len(df):,} songs")

    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved: {OUTPUT_PATH}")

    genre_counts = df["track_genre"].value_counts()
    known = genre_counts[genre_counts.index != "unknown"].sum()
    unknown = int(genre_counts.get("unknown", 0))
    print(f"\nGenre coverage: {known:,} with genre, {unknown:,} unknown")
    print("\nTop 15 genres:")
    for genre, count in genre_counts.head(15).items():
        print(f"  {genre:<35} {count:>8,}")
    print("\nDone.")


if __name__ == "__main__":
    prepare()
