from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class MusicSearchInput(BaseModel):
    song_title: str = Field(..., description="Title of the reference song")
    artist: str = Field(default="", description="Artist of the reference song (optional)")
    characteristics: str = Field(
        ...,
        description="Comma-separated musical characteristics to match (e.g. 'dark mood, fast tempo, electronic')",
    )


class MusicSimilaritySearchTool(BaseTool):
    name: str = "music_similarity_search"
    description: str = (
        "Generates targeted search queries for finding songs similar to a reference track "
        "based on its musical characteristics. Returns a list of optimised search queries "
        "you should use with a web search tool."
    )
    args_schema: Type[BaseModel] = MusicSearchInput

    def _run(self, song_title: str, characteristics: str, artist: str = "") -> str:
        artist_part = f" {artist}" if artist else ""
        traits = [t.strip() for t in characteristics.split(",") if t.strip()]
        trait_str = " ".join(traits[:3])

        queries = [
            f'songs similar to "{song_title}"{" by" + artist_part if artist_part else ""}',
            f'"{song_title}" similar artists recommendations',
            f"music like {song_title}{artist_part} recommendations",
            f"{trait_str} songs recommendations playlist",
            f'"{song_title}" fans also like',
        ]

        return "\n".join(f"- {q}" for q in queries)
