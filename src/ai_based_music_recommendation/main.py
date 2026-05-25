#!/usr/bin/env python

import contextlib
import json
import os
import re
import sys
from pathlib import Path

from ai_based_music_recommendation.crew import MusicRecommendationCrew
from ai_based_music_recommendation.tools import SongLookupTool

if sys.platform == "win32":
    os.system("")  # activeaza culorile ansi in consola windows

_W = 60

# coduri ansi pentru stilizarea textului in terminal
_R   = "\x1b[0m"
_B   = "\x1b[1m"
_DIM = "\x1b[2m"
_CY  = "\x1b[36m"
_BCY = "\x1b[96m"
_GR  = "\x1b[32m"
_BGR = "\x1b[92m"
_YL  = "\x1b[33m"
_BYL = "\x1b[93m"
_BMG = "\x1b[95m"
_WH  = "\x1b[37m"
_BWH = "\x1b[97m"


def _c(color: str, text: str) -> str:
    return f"{color}{text}{_R}"


def _sep(char: str = "─", color: str = _CY) -> str:
    return _c(color, char * _W)


# transforma raspunsul json al unui tool intr-un rezumat pe o singura linie
def _summarize_tool_output(json_str: str) -> str:
    try:
        data = json.loads(json_str)
    except Exception:
        short = json_str.strip()[:120]
        return f"[{short}{'...' if len(json_str.strip()) > 120 else ''}]"

    if isinstance(data, dict):
        if "error" in data:
            return f"[Error: {data['error']}]"
        if "track_name" in data and "features" in data:
            return (
                f"[Song: \"{data['track_name']}\" by {data.get('artists', '?')} | "
                f"Genre: {data.get('genre', '?')} | Popularity: {data.get('popularity', 0)}]"
            )
        if "top_genre_matches" in data:
            top = (data["top_genre_matches"] or [{}])[0]
            return f"[Genre: {top.get('genre', '?')} — score {top.get('similarity', 0):.2f}]"
        if "top_shared_features" in data:
            rec = data.get("recommended_song", "?")
            genre = data.get("recommended_predicted_genre", "?")
            sim = data.get("fingerprint_cosine_similarity", 0)
            n = len(data.get("top_shared_features", []))
            return (
                f"[SHAP: \"{rec}\" | genre: {genre} | "
                f"similarity: {sim:.2f} | {n} shared features]"
            )

    if (isinstance(data, list) and data
            and isinstance(data[0], dict) and "track_name" in data[0]):
        names = [f'"{s["track_name"]}"' for s in data[:2]]
        extra = f" + {len(data) - 2} more" if len(data) > 2 else ""
        return f"[Similar: {', '.join(names)}{extra}]"

    return "[data received]"


# asocierea dintre etichetele crewai si culorile lor de afisare
_LABEL_COLORS = [
    (r"^Thought:",      _B + _WH,  _WH),
    (r"^Action Input:", _CY,       _DIM),
    (r"^Action:",       _B + _BCY, _BCY),
    (r"^Observation:",  _B + _BGR, _GR),
    (r"^Final Answer:", _B + _BYL, _BWH),
]


# aplica culori pe o singura linie din outputul agentului crewai
def _colorize_line(line: str) -> str:
    s = line.strip()
    if not s:
        return line

    indent = line[: len(line) - len(line.lstrip())]

    if re.match(r"^[╭╮╰╯│─═\s]+$", s):
        return _c(_DIM, line)

    if s.startswith(("> Entering", "> Finished")):
        return _c(_DIM, line)

    clean = re.sub(r"^[╭╮╰╯│─═#\s]+", "", s)

    if clean.startswith("Agent:") and len(s) < 120:
        return re.sub(
            r"(Agent:\s*)(.+)",
            lambda m: _c(_B + _BMG, m.group(1)) + _c(_B + _BWH, m.group(2)),
            line,
        )

    if clean.startswith("Task:"):
        return _c(_DIM, line)

    for pattern, label_color, value_color in _LABEL_COLORS:
        m = re.match(pattern, s)
        if m:
            label = s[: m.end()]
            rest = s[m.end():]
            return indent + _c(label_color, label) + _c(value_color, rest)

    return line


# elimina codurile ansi brute, comprima blocurile json si recoloreaza outputul crewai
def _filter_crewai_output(text: str) -> str:
    text = re.sub(r"\x1b\[[0-9;]*[mGKHFABCDEFJKST]", "", text)

    raw = text.split("\n")
    collapsed: list[str] = []
    i = 0
    while i < len(raw):
        line = raw[i]
        obs_m = re.match(r"^(\s*Observation:\s*)(.*)", line)
        if obs_m:
            prefix, content = obs_m.group(1), obs_m.group(2).strip()
            if content and content[0] in ("{", "["):
                json_lines = [content]
                depth = (content.count("{") + content.count("[")
                         - content.count("}") - content.count("]"))
                i += 1
                while i < len(raw) and depth > 0:
                    l = raw[i]
                    json_lines.append(l)
                    depth += (l.count("{") + l.count("[")
                              - l.count("}") - l.count("]"))
                    i += 1
                collapsed.append(prefix + _summarize_tool_output("\n".join(json_lines)))
                continue
        collapsed.append(line)
        i += 1

    out: list[str] = []
    prev_blank = True
    for line in collapsed:
        s = line.strip()

        if re.match(r"^Final Answer:", s) and not prev_blank:
            out.append("")

        out.append(_colorize_line(line))

        if re.match(r"^Observation:", s):
            out.append("")

        prev_blank = s == ""

    return "\n".join(out)


# inlocuieste stdout cu un writer filtrat cat timp ruleaza crew-ul
class _FilteredWriter:
    def __init__(self, wrapped):
        self._wrapped = wrapped

    def write(self, text: str) -> int:
        self._wrapped.write(_filter_crewai_output(text))
        return len(text)

    def flush(self):
        self._wrapped.flush()

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


# context manager care redirecteaza stdout prin filtrul de culori
@contextlib.contextmanager
def _clean_console():
    original = sys.stdout
    sys.stdout = _FilteredWriter(original)
    try:
        yield
    finally:
        sys.stdout = original


# converteste markdown-ul din raportul final in text colorat pentru terminal
def _format_output(text: str) -> str:
    out: list[str] = []
    for line in text.splitlines():
        m2 = re.match(r"^## (.+)$", line)
        if m2:
            out.extend([
                "",
                _c(_B + _BCY, "═" * _W),
                _c(_B + _BWH, m2.group(1).upper()),
                _c(_B + _BCY, "═" * _W),
            ])
            continue
        m3 = re.match(r"^### (.+)$", line)
        if m3:
            out.extend(["", _c(_B + _CY, m3.group(1)), _c(_CY, "─" * _W)])
            continue
        line = re.sub(r"^(\d+\.)", lambda m: _c(_B + _BYL, m.group(1)), line)
        line = re.sub(r"\*\*(.+?)\*\*", lambda m: _c(_B + _BWH, m.group(1)), line)
        line = re.sub(r"\*(.+?)\*", r"\1", line)
        out.append(line)
    return "\n".join(out).strip()


# verifica existenta cantecului in dataset si opreste aplicatia daca nu e gasit
def _precheck(song_title: str, artist: str) -> dict:
    result = json.loads(SongLookupTool()._run(song_title=song_title, artist=artist))

    if "error" in result:
        print(_c(_BYL, f"\n  Song not found: {result['error']}"))
        suggestions = result.get("suggestions", [])
        if suggestions:
            print(_c(_YL, "\n  Did you mean one of these?"))
            for s in suggestions:
                print(f"    {_c(_CY, '-')} {s}")
        print(_c(_DIM, "\n  Tip: title and artist must match the Spotify dataset exactly."))
        sys.exit(1)

    return result


# citeste cantecul de la utilizator, lanseaza agentii si afiseaza recomandarile
def run():
    if len(sys.argv) >= 3:
        song_title = sys.argv[1]
        artist = sys.argv[2]
    else:
        print(_c(_BCY, "Song title : "), end="", flush=True)
        song_title = input().strip()
        print(_c(_BCY, "Artist     : "), end="", flush=True)
        artist = input().strip()

    if not song_title or not artist:
        print("Error: both song title and artist are required.")
        sys.exit(1)

    print()
    print(_sep("═", _BCY))
    print(_c(_B + _BCY, "  AI MUSIC RECOMMENDATIONS"))
    print(_sep("═", _BCY))
    print()

    song_data = _precheck(song_title, artist)
    print(
        f"  {_c(_B + _GR, 'Found  :')} "
        f"\"{_c(_BWH, song_data['track_name'])}\" by {_c(_BWH, song_data['artists'])}\n"
        f"  {_c(_B + _GR, 'Genre  :')} {_c(_YL, song_data['genre'])}"
        f"   {_c(_DIM, '|')}   "
        f"{_c(_B + _GR, 'Popularity:')} {_c(_YL, str(song_data['popularity']))}{_c(_DIM, '/100')}"
    )

    inputs = {"song_title": song_title, "artist": artist}
    print()
    print(_sep("─", _CY))
    print(
        f"  {_c(_YL, 'Searching for songs similar to:')} "
        f"{_c(_B + _BWH, song_title)} {_c(_DIM, '—')} {_c(_BWH, artist)}"
    )
    print(_sep("─", _CY))
    print()

    with _clean_console():
        result = MusicRecommendationCrew().crew().kickoff(inputs=inputs)

    print()
    print(_sep("═", _BCY))
    print()
    print(_format_output(result.raw))
    print()

    output_path = Path("output/recommendations.md")
    if output_path.exists():
        print()
        print(_sep("─", _CY))
        print(f"  {_c(_DIM, 'Full report saved to:')} {_c(_CY, str(output_path))}")
    print()


# antreneaza crew-ul pentru imbunatatirea performantei pe un numar de iteratii
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


# ruleaza crew-ul in mod de testare cu un model de evaluare dat
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
