#!/usr/bin/env python
"""Enrich merged_dataset.parquet with official Spotify genre labels per artist.

For every unique artist in the dataset the script calls the Spotify Search API,
retrieves the official genre list (e.g. ["pop", "dance pop", "synth-pop"]) and
stores it back in the parquet as two new columns:
    spotify_genre       — primary genre (first in Spotify's list)
    spotify_genres_all  — all genres, pipe-separated ("pop|dance pop|synth-pop")

A JSON cache at data/artist_genres_cache.json is maintained so the script can
be interrupted and resumed without re-fetching already processed artists.

Requirements:
    SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in .env
    (free account → https://developer.spotify.com/dashboard → Create App)

Usage:
    python scripts/enrich_genres_spotify.py
"""

import base64
import json
import os
import re
import sys
import time

import requests
import pandas as pd
from dotenv import load_dotenv

_ROOT      = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DATA_DIR  = os.path.join(_ROOT, "data")
MERGED_PATH = os.path.join(_DATA_DIR, "merged_dataset.parquet")
CACHE_PATH  = os.path.join(_DATA_DIR, "artist_genres_cache.json")

CHECKPOINT_EVERY = 500   # persist cache every N artists
DELAY            = 0.5   # ~2 req/s — conservative rate, minimizes ban risk
TOKEN_TTL        = 3_000 # refresh token every 50 min (token expires after 60)


# ---------------------------------------------------------------------------
# Spotify auth + search
# ---------------------------------------------------------------------------

def _get_token(client_id: str, client_secret: str) -> str:
    auth = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    r = requests.post(
        "https://accounts.spotify.com/api/token",
        headers={
            "Authorization": f"Basic {auth}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data={"grant_type": "client_credentials"},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()["access_token"]


def _primary_name(artist_field: str) -> str:
    """Extract the first/primary artist name from a possibly multi-artist string."""
    for sep in [",", ";", " & ", " feat.", " ft.", " x "]:
        if sep in artist_field:
            return artist_field.split(sep)[0].strip()
    return artist_field.strip()


def _fetch_genres(
    session: requests.Session, token: str, artist_field: str
) -> list[str] | None:
    """Return Spotify genres for artist. Returns None on auth error (token expired)."""
    primary = _primary_name(artist_field)

    r = session.get(
        "https://api.spotify.com/v1/search",
        headers={"Authorization": f"Bearer {token}"},
        params={"q": primary, "type": "artist", "limit": 1},
        timeout=15,
    )

    if r.status_code == 429:
        raw = r.headers.get("Retry-After", "5")
        try:
            wait = int(float(raw))
        except (ValueError, TypeError):
            wait = 5
        if wait > 60:
            print(
                f"\n  Spotify ban activ ({wait}s = ~{wait//3600}h {(wait%3600)//60}min)."
                "\n  Ruleaza din nou scriptul dupa expirarea ban-ului."
            )
            sys.exit(0)
        print(f"\n  [rate limit] sleeping {wait}s ...", flush=True)
        time.sleep(wait)
        return _fetch_genres(session, token, artist_field)

    if r.status_code == 401:
        return None  # signal caller to refresh token

    if r.status_code != 200:
        return []

    items = r.json().get("artists", {}).get("items", [])
    if not items:
        return []

    top = items[0]
    # Sanity-check: Spotify name should loosely match what we searched
    spotify_name = top.get("name", "").lower()
    search_name  = primary.lower()
    # Accept if either is a substring of the other (handles "The Weeknd" vs "weeknd")
    if search_name not in spotify_name and spotify_name not in search_name:
        # Try a fuzzy check: share ≥ 4 consecutive chars
        if not any(
            search_name[i:i+4] in spotify_name
            for i in range(len(search_name) - 3)
        ):
            return []

    return top.get("genres", [])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    load_dotenv(os.path.join(_ROOT, ".env"))

    client_id     = os.getenv("SPOTIFY_CLIENT_ID", "").strip()
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET", "").strip()

    if not client_id or not client_secret:
        print(
            "Missing Spotify credentials.\n"
            "1. Go to https://developer.spotify.com/dashboard\n"
            "2. Log in (free account) → Create App\n"
            "3. Copy Client ID and Client Secret into .env:\n"
            "     SPOTIFY_CLIENT_ID=<your_id>\n"
            "     SPOTIFY_CLIENT_SECRET=<your_secret>"
        )
        sys.exit(1)

    if not os.path.exists(MERGED_PATH):
        print("merged_dataset.parquet not found — run merge_datasets.py first.")
        sys.exit(1)

    print("Loading merged dataset...")
    df = pd.read_parquet(MERGED_PATH)
    print(f"Loaded {len(df):,} songs")

    unique_artists: list[str] = df["artists"].dropna().unique().tolist()
    print(f"Unique artists total: {len(unique_artists):,}")

    # ------------------------------------------------------------------
    # Load / initialise cache
    # ------------------------------------------------------------------
    cache: dict[str, list[str]] = {}
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, encoding="utf-8") as f:
            cache = json.load(f)
        print(f"Cache loaded: {len(cache):,} artists already resolved")

    # Pre-populate cache with genres already present in the dataset
    # so we skip API calls for artists whose genre is already known
    prepop = 0
    known = df[df["track_genre"] != "unknown"][["artists", "track_genre"]].dropna()
    # Build artist→genre map: first known genre per artist
    artist_genre_map = dict(
        zip(known["artists"].tolist(), known["track_genre"].tolist())
    )
    for artist, genre in artist_genre_map.items():
        if artist not in cache and genre and genre != "unknown":
            cache[artist] = [genre]
            prepop += 1
    if prepop:
        print(f"Pre-populated from dataset: {prepop:,} artists (no API call needed)")
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)

    # Only fetch artists whose songs are still all "unknown"
    remaining = [a for a in unique_artists if a not in cache]

    if remaining:
        eta_min = len(remaining) * DELAY / 60
        print(f"Artists to fetch via Spotify API: {len(remaining):,}  — ETA ≈ {eta_min:.0f} min")
        print("(script can be interrupted and resumed — cache is saved every "
              f"{CHECKPOINT_EVERY} artists)\n")

        token      = _get_token(client_id, client_secret)
        token_time = time.time()
        session    = requests.Session()
        start      = time.time()

        for i, artist in enumerate(remaining):
            # Refresh token ~50 min before expiry
            if time.time() - token_time > TOKEN_TTL:
                token      = _get_token(client_id, client_secret)
                token_time = time.time()

            genres = _fetch_genres(session, token, artist)

            if genres is None:              # token expired mid-flight
                token      = _get_token(client_id, client_secret)
                token_time = time.time()
                genres     = _fetch_genres(session, token, artist) or []

            cache[artist] = genres
            time.sleep(DELAY)

            # Progress + checkpoint
            n_done = i + 1
            if n_done % CHECKPOINT_EVERY == 0 or n_done == len(remaining):
                with open(CACHE_PATH, "w", encoding="utf-8") as f:
                    json.dump(cache, f, ensure_ascii=False)
                elapsed = time.time() - start
                rate    = n_done / elapsed
                left    = (len(remaining) - n_done) / max(rate, 0.01) / 60
                print(
                    f"  {n_done:>{len(str(len(remaining)))}}/{len(remaining)} "
                    f"| {rate:.1f} req/s | ETA {left:.0f} min",
                    flush=True,
                )
    else:
        print("All artists already resolved — applying cached genres to dataset.")

    # ------------------------------------------------------------------
    # Apply genres to the parquet
    # ------------------------------------------------------------------
    print("\nApplying genres to dataset...")

    df["spotify_genre"] = df["artists"].map(
        lambda a: (cache.get(a, []) or ["unknown"])[0]
    )
    df["spotify_genres_all"] = df["artists"].map(
        lambda a: "|".join(cache.get(a, [])) or "unknown"
    )

    df.to_parquet(MERGED_PATH, index=False)

    resolved = (df["spotify_genre"] != "unknown").sum()
    print(
        f"Saved. Genre coverage: {resolved:,}/{len(df):,} songs "
        f"({100 * resolved / len(df):.1f}%)"
    )

    print("\nTop 25 Spotify genres in dataset:")
    print(df["spotify_genre"].value_counts().head(25).to_string())


if __name__ == "__main__":
    main()
