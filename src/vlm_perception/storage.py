import asyncio
import json
from pathlib import Path

import polars as pl

from vlm_perception.models import TrialResult


def result_to_row(result: TrialResult) -> dict:
    return {
        "model": result.model,
        "prompt_id": result.prompt_id,
        "blur_px": result.condition.blur_radius,
        "crisp_on_top": result.condition.crisp_on_top,
        "crisp_side": result.condition.crisp_side.value,
        "colour_crisp": result.condition.colour_crisp.value,
        "colour_blurred": result.condition.colour_blurred.value,
        "correct_answer": result.condition.correct_answer.value,
        "parsed_answer": result.parsed_answer.value if result.parsed_answer else None,
        "correct": result.correct,
        "prompt": result.prompt,
        "raw_response": result.raw_response,
        "reasoning_trace": result.reasoning_trace,
        "timestamp": result.timestamp.isoformat(),
    }


def append_results(results: list[TrialResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for result in results:
            f.write(json.dumps(result_to_row(result)) + "\n")


async def async_append_result(
    result: TrialResult, path: Path, lock: asyncio.Lock
) -> None:
    line = json.dumps(result_to_row(result)) + "\n"
    async with lock:
        with open(path, "a") as f:
            f.write(line)


def load_results(path: Path) -> pl.DataFrame:
    return pl.read_ndjson(path)


TRIAL_KEY_COLS = [
    "model",
    "prompt_id",
    "blur_px",
    "crisp_on_top",
    "crisp_side",
    "colour_crisp",
    "colour_blurred",
]


def existing_trial_counts(path: Path) -> dict[tuple, int]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    df = load_results(path)
    counts = (
        df.group_by(TRIAL_KEY_COLS)
        .agg(pl.col("model").count().alias("n"))
    )
    result = {}
    for row in counts.iter_rows(named=True):
        key = tuple(row[col] for col in TRIAL_KEY_COLS)
        result[key] = row["n"]
    return result
