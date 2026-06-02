import json
import os
import re
from typing import Optional, Type

import numpy as np
import pandas as pd
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

# caile catre fisierele de date ale aplicatiei
_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")
)
MERGED_PATH = os.path.join(_DATA_DIR, "merged_dataset.parquet")
FINGERPRINTS_PATH = os.path.join(_DATA_DIR, "genre_fingerprints")

# caracteristicile audio numerice folosite pentru calculul similaritatii
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

# cache-uri in memorie pentru a evita reincarcarea datelor la fiecare apel
_df_cache: Optional[pd.DataFrame] = None
_norm_cache: Optional[np.ndarray] = None
_fingerprints_cache: Optional[dict] = None


# incarca datasetul din parquet, normalizeaza caracteristicile si le pastreaza in cache
def _load() -> tuple[pd.DataFrame, np.ndarray]:
    global _df_cache, _norm_cache
    if _df_cache is not None:
        return _df_cache, _norm_cache

    if not os.path.exists(MERGED_PATH):
        raise FileNotFoundError(
            f"Dataset not found at '{MERGED_PATH}'.\n"
            "Run scripts/prepare_zenodo.py first to build the dataset."
        )

    print("Loading dataset...")
    df = pd.read_parquet(MERGED_PATH)
    df = df.dropna(subset=NUMERIC_FEATURES).reset_index(drop=True)

    mat = df[NUMERIC_FEATURES].values.astype(float)
    mins, maxs = mat.min(axis=0), mat.max(axis=0)
    ranges = np.where(maxs - mins == 0, 1.0, maxs - mins)

    _df_cache = df
    _norm_cache = (mat - mins) / ranges
    print(f"Dataset ready: {len(df):,} songs.")
    return _df_cache, _norm_cache


# gaseste indexul unui cantec in dataframe dupa titlu si artist
def _find_index(df: pd.DataFrame, song_title: str, artist: str = "") -> Optional[int]:
    title_lower = song_title.lower()
    title_mask = df["track_name"].str.lower().str.contains(
        title_lower, na=False, regex=False
    )

    if artist:
        artist_mask = df["artists"].str.lower().str.contains(
            artist.lower(), na=False, regex=False
        )
        combined = title_mask & artist_mask
        mask = combined if combined.any() else title_mask
    else:
        mask = title_mask

    hits = df[mask]
    if hits.empty:
        return None

    exact = hits[hits["track_name"].str.lower() == title_lower]
    source = exact if not exact.empty else hits
    if "popularity" in source.columns:
        return int(source["popularity"].idxmax())
    return int(source.index[0])


# incarca amprenta sonora a fiecarui gen muzical din fisierul de fingerprints
def _load_fingerprints() -> dict:
    global _fingerprints_cache
    if _fingerprints_cache is not None:
        return _fingerprints_cache

    if not os.path.exists(FINGERPRINTS_PATH):
        raise FileNotFoundError(f"Genre fingerprints not found at '{FINGERPRINTS_PATH}'.")

    with open(FINGERPRINTS_PATH, "r", encoding="utf-8") as fh:
        content = fh.read()

    genre_re = re.compile(r"^## (.+)$", re.MULTILINE)
    genre_matches = list(genre_re.finditer(content))

    fingerprints: dict = {}
    for i, m in enumerate(genre_matches):
        genre_name = m.group(1).strip()
        block_start = m.end()
        block_end = genre_matches[i + 1].start() if i + 1 < len(genre_matches) else len(content)
        block = content[block_start:block_end]

        features: dict = {}
        key_signals = ""
        for line in block.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("key signals:"):
                key_signals = line[len("key signals:"):].strip()
            elif ":" in line:
                key, val = line.split(":", 1)
                val_str = val.strip().split("/")[0].strip()
                try:
                    features[key.strip()] = float(val_str)
                except ValueError:
                    pass

        if len(features) >= 9:
            vector = np.array([features.get(f, 0.0) for f in NUMERIC_FEATURES])
            fingerprints[genre_name.lower()] = {
                "name": genre_name,
                "features": features,
                "key_signals": key_signals,
                "vector": vector,
            }

    _fingerprints_cache = fingerprints
    return fingerprints


# rescaleaza caracteristicile unui cantec la scara 0-100 folosita de amprente
def _to_fingerprint_scale(row: "pd.Series | dict") -> np.ndarray:
    vec = []
    for feat in NUMERIC_FEATURES:
        val = float(row[feat]) if hasattr(row, "__getitem__") else 0.0
        if feat == "loudness":
            scaled = (val + 60.0) / 60.0 * 100.0
        elif feat == "tempo":
            scaled = val / 220.0 * 100.0
        else:
            scaled = val * 100.0
        vec.append(max(0.0, min(100.0, scaled)))
    return np.array(vec)


# prezice genul unui cantec prin comparatie cosinus cu amprentele de gen
def _predict_genre_from_fingerprints(row: "pd.Series | dict", top_k: int = 1) -> list[str]:
    try:
        fps = _load_fingerprints()
    except Exception:
        return []

    song_vec = _to_fingerprint_scale(row)
    song_norm = np.linalg.norm(song_vec) or 1.0

    scores = []
    for fp in fps.values():
        fp_norm = np.linalg.norm(fp["vector"]) or 1.0
        sim = float(np.dot(song_vec, fp["vector"]) / (song_norm * fp_norm))
        scores.append((fp["name"], sim))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in scores[:top_k]]


# extrage cuvinte cheie din numele unui gen pentru a filtra datasetul prin substring
def _genre_search_keywords(fingerprint_genre: str) -> list[str]:
    words = re.split(r"[/\s\-]+", fingerprint_genre.lower())
    return [w for w in words if len(w) > 2]


# tool pentru cautarea unui cantec in dataset si returnarea caracteristicilor sale audio
class SongLookupInput(BaseModel):
    song_title: str = Field(..., description="Title of the song to look up")
    artist: str = Field(default="", description="Artist name (optional, improves accuracy)")


class SongLookupTool(BaseTool):
    name: str = "song_lookup"
    description: str = (
        "Looks up a song in the Spotify/Zenodo dataset and returns its audio features "
        "(danceability, energy, loudness, speechiness, acousticness, instrumentalness, "
        "liveness, valence, tempo) along with genre and popularity."
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


# tool pentru identificarea genului muzical al unui cantec prin similaritate cu amprentele
class GenreFingerprinterInput(BaseModel):
    song_title: str = Field(..., description="Title of the song to classify")
    artist: str = Field(default="", description="Artist name (optional)")


class GenreFingerprinterTool(BaseTool):
    name: str = "genre_fingerprinter"
    description: str = (
        "Identifies the genre of a song by comparing its 9 audio features against "
        "30 genre fingerprint profiles using cosine similarity (nearest-centroid). "
        "Returns the top 3 matching genres with similarity scores and the key audio "
        "signals that explain the match."
    )
    args_schema: Type[BaseModel] = GenreFingerprinterInput

    def _run(self, song_title: str, artist: str = "") -> str:
        try:
            df, _ = _load()
            fps = _load_fingerprints()
        except Exception as exc:
            return json.dumps({"error": str(exc)})

        idx = _find_index(df, song_title, artist)
        if idx is None:
            return json.dumps({"error": f"'{song_title}' not found in dataset."})

        row = df.loc[idx]
        song_vec = _to_fingerprint_scale(row)
        song_norm = np.linalg.norm(song_vec) or 1.0

        matches = []
        for fp in fps.values():
            fp_norm = np.linalg.norm(fp["vector"]) or 1.0
            sim = float(np.dot(song_vec, fp["vector"]) / (song_norm * fp_norm))
            matches.append({
                "genre": fp["name"],
                "similarity": round(sim, 4),
                "key_signals": fp["key_signals"],
            })

        matches.sort(key=lambda x: x["similarity"], reverse=True)
        top3 = matches[:3]

        # comparatie caracteristica cu caracteristica fata de cel mai bun gen
        top_fp = fps[top3[0]["genre"].lower()]
        feature_gap = {}
        for feat, scaled_val in zip(NUMERIC_FEATURES, song_vec):
            fp_val = top_fp["features"].get(feat, 0.0)
            feature_gap[feat] = {
                "song_score": round(scaled_val, 1),
                "fingerprint_expected": fp_val,
                "delta": round(scaled_val - fp_val, 1),
            }

        return json.dumps(
            {
                "track_name": str(row.get("track_name", "")),
                "artists": str(row.get("artists", "")),
                "dataset_genre": str(row.get("track_genre", "")),
                "top_genre_matches": top3,
                "feature_vs_top_match": feature_gap,
            },
            indent=2,
        )


# tool pentru gasirea cantecelor similare unui cantec de referinta prin cosinus pe caracteristici
class SimilarSongSearchInput(BaseModel):
    song_title: str = Field(..., description="Title of the reference song")
    artist: str = Field(default="", description="Artist name (optional)")
    genre_filter: str = Field(default="", description="Restrict results to this genre keyword (optional)")
    top_n: int = Field(default=5, description="Number of similar songs to return")


class SimilarSongSearchTool(BaseTool):
    name: str = "similar_song_search"
    description: str = (
        "Finds the most musically similar songs to a reference track using cosine "
        "similarity on normalised audio features. Returns a ranked list with similarity "
        "scores. Genre filtering uses the fingerprint-predicted genre to keep results "
        "within the same sonic territory."
    )
    args_schema: Type[BaseModel] = SimilarSongSearchInput

    def _run(
        self,
        song_title: str,
        artist: str = "",
        genre_filter: str = "",
        top_n: int = 5,
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

        # exclude alte intrari ale aceluiasi cantec (covere/karaoke au acelasi titlu si similaritate ~1.0)
        ref_name_lower = str(df.loc[ref_idx, "track_name"]).strip().lower()
        same_title = df_work["track_name"].str.strip().str.lower() == ref_name_lower
        df_work = df_work[~same_title]

        min_candidates = top_n * 5

        if "track_genre" in df_work.columns:
            if genre_filter:
                filtered = df_work[
                    df_work["track_genre"].str.lower().str.contains(
                        genre_filter.lower(), na=False, regex=False
                    )
                ]
                if len(filtered) >= min_candidates:
                    df_work = filtered
            else:
                ref_genre = str(df.loc[ref_idx, "track_genre"])
                if ref_genre not in ("unknown", "nan", ""):
                    filtered = df_work[
                        df_work["track_genre"].str.lower().str.contains(
                            ref_genre.lower(), na=False, regex=False
                        )
                    ]
                    if len(filtered) >= min_candidates:
                        df_work = filtered
                else:
                    # genul este necunoscut, se foloseste amprenta de gen pentru filtrare
                    ref_row = df.loc[ref_idx]
                    top_fp_genres = _predict_genre_from_fingerprints(ref_row, top_k=2)
                    for k in range(1, len(top_fp_genres) + 1):
                        kws = []
                        for g in top_fp_genres[:k]:
                            kws.extend(_genre_search_keywords(g))
                        if kws:
                            pattern = "|".join(re.escape(kw) for kw in set(kws))
                            genre_mask = df_work["track_genre"].str.lower().str.contains(
                                pattern, na=False, regex=True
                            )
                            if genre_mask.sum() >= min_candidates:
                                df_work = df_work[genre_mask]
                                break

        keep_cols = ["track_name", "artists", "track_genre", "popularity", "_similarity"]
        keep_cols = [c for c in keep_cols if c in df_work.columns]
        top = df_work.nlargest(top_n, "_similarity")[keep_cols]

        results = [
            {
                "track_name": str(row["track_name"]),
                "artists": str(row["artists"]),
                "genre": str(row.get("track_genre", "")),
                "popularity": int(row.get("popularity", 0)),
                "similarity_score": round(float(row["_similarity"]), 4),
            }
            for _, row in top.iterrows()
        ]
        return json.dumps(results, indent=2)
