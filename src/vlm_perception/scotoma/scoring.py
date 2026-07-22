"""Transcription scoring for the Scotoma experiment.

The model's transcription is normalised (uppercase, letters only) and
scored against BOTH streams with two metrics: normalised Levenshtein
(robust to dropped/inserted characters) and positional Hamming
(physically meaningful, since glyph cells align position-by-position).
The bias index (d_real - d_robot) / (d_real + d_robot) lies in [-1, 1]:
+1 means the transcription matches the crisp robot stream exactly (the
exploit works), -1 means it matches the blurred real stream (the model
reads like a human), and it degrades gracefully towards 0 when the model
reads mush (both distances high).
"""

import json
import re

from pydantic import BaseModel


def normalise(text: str) -> str:
    """Uppercase and strip everything that is not an ASCII letter."""
    return "".join(c for c in text.upper() if "A" <= c <= "Z")


def levenshtein(a: str, b: str) -> int:
    """Classic edit distance (insert/delete/substitute, all cost 1)."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]


def normalised_levenshtein(a: str, b: str) -> float:
    """Levenshtein distance normalised to [0, 1] by max length."""
    n = max(len(a), len(b))
    if n == 0:
        return 0.0
    return levenshtein(a, b) / n


def positional_hamming(a: str, b: str) -> float:
    """Position-by-position mismatch rate over the longer length.

    Overhang positions (length mismatch) count as mismatches, so the
    result is in [0, 1] and comparable to normalised Levenshtein.
    """
    n = max(len(a), len(b))
    if n == 0:
        return 0.0
    mismatches = sum(x != y for x, y in zip(a, b, strict=False)) + abs(len(a) - len(b))
    return mismatches / n


def bias_index(d_real: float, d_robot: float) -> float:
    """(d_real - d_robot) / (d_real + d_robot), 0 when both distances are 0."""
    total = d_real + d_robot
    if total == 0:
        return 0.0
    return (d_real - d_robot) / total


def parse_transcriptions(raw: str) -> list[str]:
    """Extract candidate transcriptions from a model response.

    Tries, in order: a JSON object with a "text" field, a JSON object
    with a "messages" array (the dual-stream prompt), any double-quoted
    strings, then the last non-empty line. Candidates are normalised;
    empty candidates are dropped.
    """
    candidates: list[str] = []
    for match in re.finditer(r"\{[^{}]*\}", raw, re.DOTALL):
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        text = obj.get("text")
        if isinstance(text, str):
            candidates.append(text)
        messages = obj.get("messages")
        if isinstance(messages, list):
            candidates.extend(m for m in messages if isinstance(m, str))
    if not candidates:
        candidates = re.findall(r'"([^"]+)"', raw)
    if not candidates:
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        if lines:
            candidates = [lines[-1]]
    seen: set[str] = set()
    out: list[str] = []
    for c in candidates:
        n = normalise(c)
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


class TranscriptionScore(BaseModel):
    """Distances of a parsed transcription to both streams, plus bias indices.

    With multiple candidate transcriptions (the dual-stream prompt), each
    stream's distance is the minimum over candidates, so recovering both
    streams scores d_real = d_robot = 0.
    """

    raw_transcription: str | None
    dist_real_lev: float | None
    dist_robot_lev: float | None
    dist_real_ham: float | None
    dist_robot_ham: float | None
    bias_index_lev: float | None
    bias_index_ham: float | None


def score_transcription(
    raw_response: str, string_real: str, string_robot: str
) -> TranscriptionScore:
    """Parse and score a model response against both streams.

    For solo legibility trials (empty robot stream) only the real-stream
    distances are populated; the bias index needs both streams.
    """
    candidates = parse_transcriptions(raw_response)
    if not candidates:
        return TranscriptionScore(
            raw_transcription=None,
            dist_real_lev=None,
            dist_robot_lev=None,
            dist_real_ham=None,
            dist_robot_ham=None,
            bias_index_lev=None,
            bias_index_ham=None,
        )

    real = normalise(string_real)
    d_real_lev = min(normalised_levenshtein(c, real) for c in candidates)
    d_real_ham = min(positional_hamming(c, real) for c in candidates)

    if string_robot == "":
        return TranscriptionScore(
            raw_transcription="|".join(candidates),
            dist_real_lev=d_real_lev,
            dist_robot_lev=None,
            dist_real_ham=d_real_ham,
            dist_robot_ham=None,
            bias_index_lev=None,
            bias_index_ham=None,
        )

    robot = normalise(string_robot)
    d_robot_lev = min(normalised_levenshtein(c, robot) for c in candidates)
    d_robot_ham = min(positional_hamming(c, robot) for c in candidates)
    return TranscriptionScore(
        raw_transcription="|".join(candidates),
        dist_real_lev=d_real_lev,
        dist_robot_lev=d_robot_lev,
        dist_real_ham=d_real_ham,
        dist_robot_ham=d_robot_ham,
        bias_index_lev=bias_index(d_real_lev, d_robot_lev),
        bias_index_ham=bias_index(d_real_ham, d_robot_ham),
    )
