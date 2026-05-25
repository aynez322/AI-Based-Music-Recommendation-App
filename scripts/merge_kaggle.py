#!/usr/bin/env python
# combina cele 8 fisiere csv kaggle intr-un singur fisier parquet de antrenament
import os
import sys

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
DATASETS_DIR = os.path.join(DATA_DIR, "datasets")
OUTPUT_PATH = os.path.join(DATA_DIR, "training_dataset.parquet")

NUMERIC_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
]
OUTPUT_COLS = (
    ["track_name", "artists", "track_genre", "popularity", "key", "mode"]
    + NUMERIC_FEATURES
)

# corespondenta intre notele muzicale si numerele de clasa de inaltime spotify
PITCH_TO_INT: dict[str, int] = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}


def _p(filename: str) -> str:
    return os.path.join(DATASETS_DIR, filename)


# returneaza primul token dintr-un sir delimitat
def _first_token(value, sep: str = ",") -> str:
    if not isinstance(value, str) or not value.strip():
        return "unknown"
    first = value.split(sep)[0].strip().strip("'\"[]")
    return first if first else "unknown"


# converteste o nota muzicala sau un numar in indexul de clasa de inaltime spotify
def _pitch_to_int(value) -> int:
    if isinstance(value, str):
        return PITCH_TO_INT.get(value.strip(), -1)
    try:
        v = int(float(value))
        return v if 0 <= v <= 11 else -1
    except (TypeError, ValueError):
        return -1


def _to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


# normalizeaza tipurile de date, completeaza valorile lipsa si eticheteaza sursa
def _finalize(df: pd.DataFrame, source: str) -> pd.DataFrame:
    df = df.copy()
    df["track_name"] = df["track_name"].fillna("").astype(str).str.strip()
    df["artists"]    = df["artists"].fillna("").astype(str).str.strip()
    df["track_genre"] = df["track_genre"].fillna("unknown").astype(str).str.strip()
    df["track_genre"] = df["track_genre"].replace({"": "unknown", "nan": "unknown"})
    if "popularity" in df.columns:
        df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0).astype(int)
    else:
        df["popularity"] = 0
    df["key"]  = df["key"].fillna(-1).astype(int)
    df["mode"] = df["mode"].fillna(-1).astype(int)
    for feat in NUMERIC_FEATURES:
        df[feat] = pd.to_numeric(df[feat], errors="coerce")
    df = df.dropna(subset=NUMERIC_FEATURES).reset_index(drop=True)
    df["_source"] = source
    keep = [c for c in OUTPUT_COLS if c in df.columns] + ["_source"]
    return df[keep]


# incarca classichit.csv care foloseste coloane cu majuscule si chei ca numere intregi
def load_classichit() -> pd.DataFrame:
    df = pd.read_csv(_p("ClassicHit.csv"), low_memory=False, encoding="utf-8", encoding_errors="replace")
    df = df.rename(columns={"Track": "track_name", "Artist": "artists", "Genre": "track_genre"})
    feat_rename = {f.capitalize(): f for f in NUMERIC_FEATURES}
    feat_rename.update({
        "Danceability": "danceability", "Energy": "energy", "Loudness": "loudness",
        "Speechiness": "speechiness", "Acousticness": "acousticness",
        "Instrumentalness": "instrumentalness", "Liveness": "liveness",
        "Valence": "valence", "Tempo": "tempo",
        "Key": "key", "Mode": "mode", "Popularity": "popularity",
    })
    df = df.rename(columns=feat_rename)
    df["key"]  = df["key"].apply(_pitch_to_int) if df["key"].dtype == object else df["key"].fillna(-1).astype(int)
    df["mode"] = pd.to_numeric(df["mode"], errors="coerce").fillna(-1).astype(int) if "mode" in df.columns else pd.Series([-1] * len(df))
    return _finalize(df, "ClassicHit")


# incarca dataset.csv cu schema standard spotify
def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(_p("dataset.csv"), low_memory=False, encoding="utf-8", encoding_errors="replace")
    df["key"]  = df["key"].apply(_pitch_to_int) if df["key"].dtype == object else df["key"].fillna(-1).astype(int)
    df["mode"] = pd.to_numeric(df["mode"], errors="coerce").fillna(-1).astype(int) if "mode" in df.columns else pd.Series([-1] * len(df))
    return _finalize(df, "dataset")


# incarca music info.csv unde genul este adesea gol si se foloseste primul tag ca fallback
def load_music_info() -> pd.DataFrame:
    df = pd.read_csv(_p("Music Info.csv"), low_memory=False, encoding="utf-8", encoding_errors="replace")
    df = df.rename(columns={"name": "track_name", "artist": "artists"})
    def _genre(row):
        g = str(row.get("genre", "")).strip()
        if g and g not in ("", "nan"):
            return g
        return _first_token(row.get("tags", ""), sep=",")
    df["track_genre"] = df.apply(_genre, axis=1)
    df["key"]  = df["key"].apply(_pitch_to_int) if df["key"].dtype == object else df["key"].fillna(-1).astype(int)
    df["mode"] = pd.to_numeric(df["mode"], errors="coerce").fillna(-1).astype(int) if "mode" in df.columns else pd.Series([-1] * len(df))
    return _finalize(df, "MusicInfo")


# incarca spotifyfeatures.csv unde cheia este o nota muzicala si modul este major sau minor
def load_spotify_features() -> pd.DataFrame:
    df = pd.read_csv(_p("SpotifyFeatures.csv"), low_memory=False, encoding="utf-8", encoding_errors="replace")
    df = df.rename(columns={"artist_name": "artists", "track_name": "track_name"})
    df["track_genre"] = df["genre"].fillna("unknown")
    df["key"]  = df["key"].apply(_pitch_to_int)
    df["mode"] = df["mode"].map({"Major": 1, "Minor": 0}).fillna(-1).astype(int)
    return _finalize(df, "SpotifyFeatures")


# incarca fisierele top_10000 unde caracteristicile sunt siruri cu ghilimele
def load_top10000(filename: str) -> pd.DataFrame:
    df = pd.read_csv(_p(filename), low_memory=False, encoding="utf-8", encoding_errors="replace")
    df = df.rename(columns={
        "Track Name":      "track_name",
        "Artist Name(s)":  "artists",
        "Artist Genres":   "track_genre",
        "Popularity":      "popularity",
    })
    df["track_genre"] = df["track_genre"].apply(lambda v: _first_token(v, sep=","))
    feat_rename = {
        "Danceability": "danceability", "Energy": "energy", "Loudness": "loudness",
        "Speechiness": "speechiness", "Acousticness": "acousticness",
        "Instrumentalness": "instrumentalness", "Liveness": "liveness",
        "Valence": "valence", "Tempo": "tempo", "Key": "key", "Mode": "mode",
    }
    df = df.rename(columns=feat_rename)
    for col in NUMERIC_FEATURES + ["key", "mode", "popularity"]:
        if col in df.columns:
            df[col] = _to_float(df[col])
    df["key"]  = df["key"].fillna(-1).astype(int)
    df["mode"] = df["mode"].fillna(-1).astype(int)
    source = "top10000_1950" if "1950" in filename else "top10000_1960"
    return _finalize(df, source)


# incarca train.csv cu schema standard (continut similar cu dataset.csv, dedup elimina duplicatele)
def load_train() -> pd.DataFrame:
    df = pd.read_csv(_p("train.csv"), low_memory=False, encoding="utf-8", encoding_errors="replace")
    df["key"]  = df["key"].apply(_pitch_to_int) if df["key"].dtype == object else df["key"].fillna(-1).astype(int)
    df["mode"] = pd.to_numeric(df["mode"], errors="coerce").fillna(-1).astype(int) if "mode" in df.columns else pd.Series([-1] * len(df))
    return _finalize(df, "train")


# combina toate datasetele kaggle, elimina duplicatele si salveaza rezultatul
def merge() -> None:
    loaders = [
        ("ClassicHit.csv",          load_classichit),
        ("dataset.csv",             load_dataset),
        ("Music Info.csv",          load_music_info),
        ("SpotifyFeatures.csv",     load_spotify_features),
        ("top_10000_1950-now.csv",  lambda: load_top10000("top_10000_1950-now.csv")),
        ("top_10000_1960-now.csv",  lambda: load_top10000("top_10000_1960-now.csv")),
        ("train.csv",               load_train),
    ]
    skipped = ["spotify_songs_dataset.csv"]
    print(f"Skipping (no audio features): {skipped}\n")

    frames: list[pd.DataFrame] = []
    for filename, loader in loaders:
        path = _p(filename)
        if not os.path.exists(path):
            print(f"  [MISSING] {filename} — skipping")
            continue
        print(f"Loading {filename}...")
        try:
            df = loader()
            print(f"  → {len(df):>7,} valid rows | genres: {df['track_genre'].nunique()} unique")
            frames.append(df)
        except Exception as exc:
            print(f"  [ERROR] {filename}: {exc}")

    if not frames:
        print("No data loaded.")
        sys.exit(1)

    print(f"\nCombining {len(frames)} datasets...")
    combined = pd.concat(frames, ignore_index=True)
    print(f"  Total before dedup: {len(combined):,}")

    # elimina genurile care nu apartin unui sistem de recomandare muzicala
    DROP_GENRES = {"comedy", "anime", "soundtrack", "opera", "movie"}
    before_drop = len(combined)
    combined = combined[
        ~combined["track_genre"].str.lower().str.strip().isin(DROP_GENRES)
    ].reset_index(drop=True)
    print(f"  Dropped {before_drop - len(combined):,} rows (Comedy/Anime/Soundtrack/Opera/Movie)")

    # deduplicare dupa (track_name, artists), se pastreaza cel mai popular cu gen cunoscut
    combined["_key"] = (
        combined["track_name"].str.lower().str.strip()
        + "|||"
        + combined["artists"].str.lower().str.strip()
    )
    combined["_has_genre"] = (combined["track_genre"] != "unknown").astype(int)
    combined = (
        combined
        .sort_values(["_has_genre", "popularity"], ascending=[False, False])
        .drop_duplicates(subset="_key", keep="first")
        .drop(columns=["_key", "_has_genre"])
        .reset_index(drop=True)
    )
    print(f"  Total after dedup:  {len(combined):,}")

    print("\nRows per source (after dedup):")
    for src, cnt in combined["_source"].value_counts().items():
        print(f"  {src:<22} {cnt:>8,}")

    combined = combined.drop(columns=["_source"])

    genre_counts = combined["track_genre"].value_counts()
    known = genre_counts[genre_counts.index != "unknown"].sum()
    unknown = int(genre_counts.get("unknown", 0))
    print(f"\nGenre coverage: {known:,} labelled, {unknown:,} unknown")
    print("\nTop 20 genres:")
    for genre, cnt in genre_counts.head(20).items():
        bar = "█" * min(40, int(cnt / max(genre_counts) * 40))
        print(f"  {genre:<35} {cnt:>7,}  {bar}")

    combined.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved → {OUTPUT_PATH}")
    print(f"Columns: {list(combined.columns)}")
    print(f"Total songs: {len(combined):,}")
    print("\nDone.")


if __name__ == "__main__":
    merge()
