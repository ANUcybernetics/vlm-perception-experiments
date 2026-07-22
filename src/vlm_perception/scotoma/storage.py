"""JSONL storage for Scotoma transcription trials (separate from circles)."""

import asyncio
import json
from pathlib import Path

import polars as pl

from vlm_perception.scotoma.experiment import ScotomaTrialResult


def result_to_row(result: ScotomaTrialResult) -> dict:
    c = result.condition
    return {
        "model": result.model,
        "prompt_id": result.prompt_id,
        "trial_type": "legibility" if c.is_solo else "main",
        "pool": c.pool,
        "string_real": c.string_real,
        "string_robot": c.string_robot,
        "blur_fraction": c.blur_fraction,
        "blur_px": c.blur_px,
        "offset_fraction": c.offset_fraction,
        "font_size": c.font_size,
        "blurred_on_top": c.blurred_on_top,
        "colour_real": c.colour_real.value,
        "colour_robot": c.colour_robot.value,
        "d_pair": c.d_pair,
        "raw_transcription": result.raw_transcription,
        "dist_real_lev": result.dist_real_lev,
        "dist_robot_lev": result.dist_robot_lev,
        "dist_real_ham": result.dist_real_ham,
        "dist_robot_ham": result.dist_robot_ham,
        "bias_index_lev": result.bias_index_lev,
        "bias_index_ham": result.bias_index_ham,
        "prompt": result.prompt,
        "raw_response": result.raw_response,
        "reasoning_trace": result.reasoning_trace,
        "timestamp": result.timestamp.isoformat(),
    }


async def async_append_result(
    result: ScotomaTrialResult, path: Path, lock: asyncio.Lock
) -> None:
    line = json.dumps(result_to_row(result)) + "\n"
    async with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(line)


def load_results(path: Path) -> pl.DataFrame:
    # Large infer_schema_length so sparse fields (reasoning_trace, the
    # robot-stream distances absent from legibility trials) are detected
    # regardless of row order.
    return pl.read_ndjson(path, infer_schema_length=20000)


TRIAL_KEY_COLS = [
    "model",
    "prompt_id",
    "string_real",
    "string_robot",
    "blur_fraction",
    "blurred_on_top",
    "colour_real",
    "font_size",
]


def existing_trial_counts(path: Path) -> dict[tuple, int]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    df = load_results(path)
    counts = df.group_by(TRIAL_KEY_COLS).agg(pl.col("model").count().alias("n"))
    result = {}
    for row in counts.iter_rows(named=True):
        key = tuple(row[col] for col in TRIAL_KEY_COLS)
        result[key] = row["n"]
    return result
