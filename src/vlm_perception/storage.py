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
    rows = [result_to_row(r) for r in results]
    new_df = pl.DataFrame(rows)
    if path.exists():
        existing = pl.read_csv(path)
        combined = pl.concat([existing, new_df], how="diagonal")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        combined = new_df
    combined.write_csv(path)


def load_results(path: Path) -> pl.DataFrame:
    return pl.read_csv(path)
