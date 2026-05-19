import json
import os
import re
from typing import Optional, Type

import numpy as np
import pandas as pd
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")
)
RAW_CSV_PATH   = os.path.join(_DATA_DIR, "songs_with_attributes_and_lyrics.csv")
MERGED_PATH    = os.path.join(_DATA_DIR, "merged_dataset.parquet")
PROCESSED_PATH = os.path.join(_DATA_DIR, "processed_dataset.parquet")

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

_df_cache: Optional[pd.DataFrame] = None
_norm_cache: Optional[np.ndarray] = None

def _pick(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    missing = []

    for std, candidates in [
        ("track_name",  _TITLE_CANDIDATES),
        ("artists",     _ARTIST_CANDIDATES),
    ]:
        col = _pick(df, candidates)
        if col is None:
            missing.append(std)
        elif col != std:
            rename[col] = std

    if missing:
        raise RuntimeError(
            f"Could not detect required columns {missing} in dataset.\n"
            f"Available columns: {df.columns.tolist()}"
        )

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

    return df

def _build_processed() -> pd.DataFrame:
    if not os.path.exists(RAW_CSV_PATH):
        raise FileNotFoundError(
            f"Dataset not found at '{RAW_CSV_PATH}'.\n"
            "Place 'songs_with_attributes_and_lyrics.csv' in the data/ folder."
        )

    print(f"Loading {os.path.basename(RAW_CSV_PATH)}  (this may take ~30s for 1M rows)...")
    df = pd.read_csv(RAW_CSV_PATH, low_memory=False)
    print(f"Raw dataset: {len(df):,} rows, {len(df.columns)} columns")

    df = _normalise_columns(df)
    df = df.dropna(subset=NUMERIC_FEATURES).reset_index(drop=True)

    keep = ["track_name", "artists", "track_genre", "popularity"] + NUMERIC_FEATURES
    df = df[[c for c in keep if c in df.columns]].copy()

    os.makedirs(_DATA_DIR, exist_ok=True)
    df.to_parquet(PROCESSED_PATH, index=False)
    print(f"Processed: {len(df):,} songs saved to processed_dataset.parquet")
    return df

def _load() -> tuple[pd.DataFrame, np.ndarray]:
    global _df_cache, _norm_cache
    if _df_cache is not None:
        return _df_cache, _norm_cache

    if os.path.exists(MERGED_PATH):
        print("Loading merged dataset from cache...")
        df = pd.read_parquet(MERGED_PATH)
        df = df.dropna(subset=NUMERIC_FEATURES).reset_index(drop=True)
    elif os.path.exists(PROCESSED_PATH):
        print("Loading dataset from cache...")
        df = pd.read_parquet(PROCESSED_PATH)
        df = df.dropna(subset=NUMERIC_FEATURES).reset_index(drop=True)
    else:
        print("First run — downloading and processing dataset...")
        df = _build_processed()

    mat = df[NUMERIC_FEATURES].values.astype(float)
    mins, maxs = mat.min(axis=0), mat.max(axis=0)
    ranges = np.where(maxs - mins == 0, 1.0, maxs - mins)

    _df_cache = df
    _norm_cache = (mat - mins) / ranges
    print(f"Dataset ready: {len(df):,} songs.")
    return _df_cache, _norm_cache


def _find_index(df: pd.DataFrame, song_title: str, artist: str = "") -> Optional[int]:
    title_lower = song_title.lower()
    mask = df["track_name"].str.lower().str.contains(title_lower, na=False, regex=False)
    if artist:
        mask &= df["artists"].str.lower().str.contains(
            artist.lower(), na=False, regex=False
        )
    hits = df[mask]
    if hits.empty:
        return None
    exact = hits[hits["track_name"].str.lower() == title_lower]
    source = exact if not exact.empty else hits
    if "popularity" in source.columns:
        return int(source["popularity"].idxmax())
    return int(source.index[0])


class SongLookupInput(BaseModel):
    song_title: str = Field(..., description="Title of the song to look up")
    artist: str = Field(default="", description="Artist name (optional, improves accuracy)")


class SongLookupTool(BaseTool):
    name: str = "song_lookup"
    description: str = (
        "Looks up a song in the Spotify dataset (~1M songs) and returns its "
        "audio features (danceability, energy, tempo, valence, genre, etc.)."
    )
    args_schema: Type[BaseModel] = SongLookupInput

    def _run(self, song_title: str, artist: str = "") -> str:
        try:
            df, _ = _load()
        except Exception as exc:
            return json.dumps({"error": str(exc)})

        idx = _find_index(df, song_title, artist)
        if idx is None:
            mask = df["track_name"].str.lower().str.contains(
                song_title.lower()[:5], na=False, regex=False
            )
            suggestions = df[mask]["track_name"].unique()[:5].tolist()
            return json.dumps(
                {"error": f"'{song_title}' not found.", "suggestions": suggestions}
            )

        row = df.loc[idx]
        return json.dumps(
            {
                "track_name": str(row.get("track_name", "")),
                "artists": str(row.get("artists", "")),
                "genre": str(row.get("track_genre", "")),
                "popularity": int(row.get("popularity", 0)),
                "features": {
                    feat: round(float(row[feat]), 4)
                    for feat in NUMERIC_FEATURES
                    if feat in row
                },
            },
            indent=2,
        )


def _predict_top_genres(features: np.ndarray, top_k: int = 3) -> list:
    """Return top-k predicted genre names for a 1-D feature vector."""
    try:
        from ai_based_music_recommendation.tools.explainability_tools import (
            _get_model_and_explainer,
        )
        model, le, _ = _get_model_and_explainer()
        proba = model.predict_proba(features.reshape(1, -1))[0]
        top_indices = np.argsort(proba)[::-1][:top_k]
        return le.inverse_transform(top_indices).tolist()
    except Exception:
        return []


class SimilarSongSearchInput(BaseModel):
    song_title: str = Field(..., description="Title of the reference song")
    artist: str = Field(default="", description="Artist name (optional)")
    genre_filter: str = Field(default="", description="Restrict results to this genre (optional)")
    top_n: int = Field(default=10, description="Number of similar songs to return")


class SimilarSongSearchTool(BaseTool):
    name: str = "similar_song_search"
    description: str = (
        "Finds the most musically similar songs to a reference track using cosine "
        "similarity on normalised audio features across ~1M Spotify songs. "
        "Returns a ranked list with similarity scores."
    )
    args_schema: Type[BaseModel] = SimilarSongSearchInput

    def _run(
        self,
        song_title: str,
        artist: str = "",
        genre_filter: str = "",
        top_n: int = 10,
    ) -> str:
        try:
            df, norm = _load()
        except Exception as exc:
            return json.dumps({"error": str(exc)})

        ref_idx = _find_index(df, song_title, artist)
        if ref_idx is None:
            return json.dumps({"error": f"'{song_title}' not found in dataset."})

        ref_vec = norm[ref_idx]
        ref_norm = np.linalg.norm(ref_vec) or 1.0
        row_norms = np.linalg.norm(norm, axis=1)
        row_norms = np.where(row_norms == 0, 1.0, row_norms)
        similarities = (norm @ ref_vec) / (row_norms * ref_norm)

        df_work = df.copy()
        df_work["_similarity"] = similarities
        df_work = df_work[df_work.index != ref_idx]

        genre_col = (
            "predicted_genre" if "predicted_genre" in df_work.columns else "track_genre"
        )

        min_candidates = top_n * 5

        if genre_filter:
            for col in ("spotify_genres_all", "predicted_genre", "track_genre"):
                if col not in df_work.columns:
                    continue
                filtered = df_work[
                    df_work[col].str.lower().str.contains(
                        genre_filter.lower(), na=False, regex=False
                    )
                ]
                if not filtered.empty:
                    df_work = filtered
                    break

        elif "spotify_genres_all" in df_work.columns:
            ref_genres_str = str(df.loc[ref_idx, "spotify_genres_all"])
            if ref_genres_str != "unknown":
                ref_genres = [g for g in ref_genres_str.split("|") if g]
                pattern = "|".join(re.escape(g) for g in ref_genres)
                genre_mask = df_work["spotify_genres_all"].str.contains(
                    pattern, na=False, regex=True
                )
                if genre_mask.sum() >= min_candidates:
                    df_work = df_work[genre_mask]

        elif genre_col in df_work.columns:
            ref_features = df.loc[ref_idx, NUMERIC_FEATURES].values.astype(float)
            top_genres = _predict_top_genres(ref_features, top_k=3)
            applied = []
            for genre in top_genres:
                applied.append(genre)
                genre_mask = df_work[genre_col].isin(applied)
                if genre_mask.sum() >= min_candidates:
                    df_work = df_work[genre_mask]
                    break

        keep_cols = [
            "track_name", "artists", "track_genre", "predicted_genre",
            "popularity", "_similarity",
        ]
        keep_cols = [c for c in keep_cols if c in df_work.columns]
        top = df_work.nlargest(top_n, "_similarity")[keep_cols]

        results = [
            {
                "track_name": str(row["track_name"]),
                "artists": str(row["artists"]),
                "genre": str(row.get("predicted_genre", row.get("track_genre", ""))),
                "popularity": int(row.get("popularity", 0)),
                "similarity_score": round(float(row["_similarity"]), 4),
            }
            for _, row in top.iterrows()
        ]
        return json.dumps(results, indent=2)
