import asyncio

from vlm_perception.scotoma.evaluate import _make_trial_result
from vlm_perception.scotoma.experiment import (
    legibility_conditions,
    scotoma_conditions,
)
from vlm_perception.scotoma.storage import (
    TRIAL_KEY_COLS,
    async_append_result,
    existing_trial_counts,
    load_results,
    result_to_row,
)

SCHEMA_FIELDS = {
    "model",
    "prompt_id",
    "trial_type",
    "pool",
    "string_real",
    "string_robot",
    "blur_fraction",
    "blur_px",
    "offset_fraction",
    "font_size",
    "blurred_on_top",
    "colour_real",
    "colour_robot",
    "d_pair",
    "raw_transcription",
    "dist_real_lev",
    "dist_robot_lev",
    "dist_real_ham",
    "dist_robot_ham",
    "bias_index_lev",
    "bias_index_ham",
    "prompt",
    "raw_response",
    "reasoning_trace",
    "timestamp",
}


def _result(condition, raw='{"text": "BACKGROUND"}'):
    return _make_trial_result(raw, condition, "test-model", "naive", "prompt text")


def test_result_to_row_schema():
    row = result_to_row(_result(scotoma_conditions()[0]))
    assert set(row) == SCHEMA_FIELDS
    assert row["trial_type"] == "main"
    assert row["d_pair"] >= 8


def test_legibility_row_marked():
    row = result_to_row(_result(legibility_conditions()[0]))
    assert row["trial_type"] == "legibility"
    assert row["d_pair"] is None
    assert row["bias_index_lev"] is None


def test_append_and_resume_counts(tmp_path):
    path = tmp_path / "scotoma.jsonl"
    lock = asyncio.Lock()
    conditions = scotoma_conditions()[:3]

    async def write_all():
        for c in conditions:
            await async_append_result(_result(c), path, lock)
        # one condition evaluated twice
        await async_append_result(_result(conditions[0]), path, lock)

    asyncio.run(write_all())

    df = load_results(path)
    assert len(df) == 4

    counts = existing_trial_counts(path)
    key0 = tuple(result_to_row(_result(conditions[0]))[col] for col in TRIAL_KEY_COLS)
    assert counts[key0] == 2
    assert sum(counts.values()) == 4


def test_existing_trial_counts_missing_file(tmp_path):
    assert existing_trial_counts(tmp_path / "nope.jsonl") == {}
