import asyncio
import json
from pathlib import Path

import polars as pl

from vlm_perception.models import TrialResult

COLUMNS = [
    "model",
    "prompt_id",
    "crisp_on_top",
    "crisp_side",
    "colour_crisp",
    "colour_blurred",
    "correct_answer",
    "parsed_answer",
    "correct",
    "prompt",
    "raw_response",
    "timestamp",
]


def result_to_row(result: TrialResult) -> dict:
    return {
        "model": result.model,
        "prompt_id": result.prompt_id,
        "crisp_on_top": result.condition.crisp_on_top,
        "crisp_side": result.condition.crisp_side.value,
        "colour_crisp": result.condition.colour_crisp.value,
        "colour_blurred": result.condition.colour_blurred.value,
        "correct_answer": result.condition.correct_answer.value,
        "parsed_answer": result.parsed_answer.value if result.parsed_answer else None,
        "correct": result.correct,
        "prompt": result.prompt,
        "raw_response": result.raw_response,
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
