#!/usr/bin/env python

import json
import sys
from pathlib import Path

from ai_based_music_recommendation.crew import MusicRecommendationCrew
from ai_based_music_recommendation.tools import SongLookupTool


def _precheck(song_title: str, artist: str) -> dict:
    print("Checking dataset...")
    result = json.loads(SongLookupTool()._run(song_title=song_title, artist=artist))

    if "error" in result:
        print(f"\nSong not found: {result['error']}")
        suggestions = result.get("suggestions", [])
        if suggestions:
            print("\nDid you mean one of these titles?")
            for s in suggestions:
                print(f"  - {s}")
        print("\nTip: make sure the title and artist match the Spotify dataset exactly.")
        sys.exit(1)

    return result


def run():
    if len(sys.argv) >= 3:
        song_title = sys.argv[1]
        artist = sys.argv[2]
    else:
        song_title = input("Enter song title: ").strip()
        artist = input("Enter artist name: ").strip()

    if not song_title or not artist:
        print("Error: both song title and artist are required.")
        sys.exit(1)

    # Pre-check: abort early if song is not in the dataset
    song_data = _precheck(song_title, artist)
    print(
        f"Found: \"{song_data['track_name']}\" by {song_data['artists']}"
        f" — genre: {song_data['genre']}, popularity: {song_data['popularity']}"
    )

    inputs = {"song_title": song_title, "artist": artist}
    print(f"\nSearching for songs similar to '{song_title}' by {artist}...\n")
    print("=" * 60)

    result = MusicRecommendationCrew().crew().kickoff(inputs=inputs)

    print("\n" + "=" * 60)
    print("FINAL RECOMMENDATIONS")
    print("=" * 60)
    print(result.raw)

    output_path = Path("output/recommendations.md")
    if output_path.exists():
        print(f"\nFull report saved to: {output_path}")


def train():
    inputs = {"song_title": "Bohemian Rhapsody", "artist": "Queen"}
    try:
        MusicRecommendationCrew().crew().train(
            n_iterations=int(sys.argv[1]),
            filename=sys.argv[2],
            inputs=inputs,
        )
    except Exception as e:
        raise Exception(f"Error training crew: {e}") from e


def test():
    inputs = {"song_title": "Bohemian Rhapsody", "artist": "Queen"}
    try:
        MusicRecommendationCrew().crew().test(
            n_iterations=int(sys.argv[1]),
            eval_llm=sys.argv[2],
            inputs=inputs,
        )
    except Exception as e:
        raise Exception(f"Error testing crew: {e}") from e


if __name__ == "__main__":
    run()
