#!/usr/bin/env python
import argparse
import os
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Column detection — mirrors dataset_tools.py
# ---------------------------------------------------------------------------

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

_TITLE_CANDIDATES  = ["track_name", "Track Name", "name", "title", "song_name", "Song Name", "track"]
_ARTIST_CANDIDATES = ["artists", "artist", "Artist", "Artist Name(s)", "Artist Names", "performer", "track_artist"]
_GENRE_CANDIDATES  = ["track_genre", "genre", "Genre", "playlist_genre", "top genre"]
_POP_CANDIDATES    = ["popularity", "Popularity", "track_popularity", "streams"]

OUTPUT_COLUMNS = ["track_name", "artists", "track_genre", "popularity"] + NUMERIC_FEATURES

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))


def _pick(df: pd.DataFrame, candidates: list) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _normalise(df: pd.DataFrame, source_label: str) -> pd.DataFrame | None:
    """Rename columns to standard schema and drop rows missing numeric features."""
    rename = {}
    missing = []

    for std, candidates in [
        ("track_name", _TITLE_CANDIDATES),
        ("artists",    _ARTIST_CANDIDATES),
    ]:
        col = _pick(df, candidates)
        if col is None:
            missing.append(std)
        elif col != std:
            rename[col] = std

    if missing:
        print(f"  [SKIP] {source_label}: could not detect required columns {missing}")
        print(f"         Available columns: {df.columns.tolist()[:20]}")
        return None

    for std, candidates in [
        ("track_genre", _GENRE_CANDIDATES),
        ("popularity",  _POP_CANDIDATES),
    ]:
        col = _pick(df, candidates)
        if col and col != std:
            rename[col] = std

    df = df.rename(columns=rename)

    if "track_genre" not in df.columns:
        df["track_genre"] = "unknown"
    if "popularity" not in df.columns:
        df["popularity"] = 0

    keep = [c for c in OUTPUT_COLUMNS if c in df.columns]
    missing_numeric = [f for f in NUMERIC_FEATURES if f not in df.columns]
    if missing_numeric:
        print(f"  [SKIP] {source_label}: missing numeric features {missing_numeric}")
        return None

    df = df[keep].copy()
    before = len(df)
    df = df.dropna(subset=NUMERIC_FEATURES).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped:,} rows with missing numeric features")

    df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0).astype(int)

    return df


def _find_csvs(data_dir: str) -> list[str]:
    paths = []
    for root, _, files in os.walk(data_dir):
        for f in sorted(files):
            if f.endswith(".csv"):
                paths.append(os.path.join(root, f))
    return paths


def merge(export_csv: bool = False) -> None:
    csv_files = _find_csvs(DATA_DIR)
    if not csv_files:
        print(f"No CSV files found under {DATA_DIR}")
        sys.exit(1)

    print(f"Found {len(csv_files)} CSV file(s):\n")
    frames = []

    for path in csv_files:
        label = os.path.relpath(path, DATA_DIR)
        print(f"  Loading {label}...")
        try:
            df_raw = pd.read_csv(path, low_memory=False)
        except Exception as exc:
            print(f"  [ERROR] Could not read {label}: {exc}")
            continue

        print(f"  {len(df_raw):,} rows, {len(df_raw.columns)} columns")
        df = _normalise(df_raw, label)
        if df is None:
            continue

        df["_source"] = label
        print(f"  Kept {len(df):,} valid rows")
        frames.append(df)

    if not frames:
        print("\nNo datasets could be loaded — nothing to merge.")
        sys.exit(1)

    print(f"\nMerging {len(frames)} dataset(s)...")
    combined = pd.concat(frames, ignore_index=True)
    print(f"Total rows before dedup: {len(combined):,}")

    combined["_key"] = (
        combined["track_name"].str.strip().str.lower()
        + "|||"
        + combined["artists"].str.strip().str.lower()
    )

    combined["_has_genre"] = (combined["track_genre"] != "unknown").astype(int)
    combined = combined.sort_values(
        ["_has_genre", "popularity"], ascending=[False, False]
    )
    combined = combined.drop_duplicates(subset="_key", keep="first")
    combined = combined.drop(columns=["_key", "_has_genre", "_source"])
    combined = combined.reset_index(drop=True)

    print(f"Total rows after dedup:  {len(combined):,}")

    out_parquet = os.path.join(DATA_DIR, "merged_dataset.parquet")
    combined.to_parquet(out_parquet, index=False)
    print(f"\nSaved: {out_parquet}")

    if export_csv:
        out_csv = os.path.join(DATA_DIR, "merged_dataset.csv")
        combined.to_csv(out_csv, index=False)
        print(f"Saved: {out_csv}")

    print(f"\nColumn summary:")
    for col in OUTPUT_COLUMNS:
        if col in combined.columns:
            non_null = combined[col].notna().sum()
            print(f"  {col:<22} {non_null:>10,} non-null values")

    if "track_genre" in combined.columns:
        genres = combined["track_genre"].value_counts()
        known = genres[genres.index != "unknown"].sum()
        unknown = genres.get("unknown", 0)
        print(f"\nGenre coverage: {known:,} with genre labels, {unknown:,} unknown")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge Spotify CSV datasets")
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Also export merged_dataset.csv alongside the parquet",
    )
    args = parser.parse_args()
    merge(export_csv=args.csv)
