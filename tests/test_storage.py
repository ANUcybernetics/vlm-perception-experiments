from datetime import UTC, datetime

from vlm_perception.models import Colour, Condition, Side, TrialResult
from vlm_perception.storage import append_results, load_results, result_to_row


def _make_result(correct: bool | None = True, model: str = "test-model") -> TrialResult:
    return TrialResult(
        condition=Condition(
            crisp_on_top=True,
            crisp_side=Side.left,
            colour_crisp=Colour.red,
            colour_blurred=Colour.blue,
        ),
        model=model,
        prompt_id="neutral",
        prompt="test prompt",
        raw_response='{"answer": "left"}',
        parsed_answer=Side.left,
        correct=correct,
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_result_to_row_fields():
    row = result_to_row(_make_result())
    assert row["model"] == "test-model"
    assert row["crisp_on_top"] is True
    assert row["crisp_side"] == "left"
    assert row["colour_crisp"] == "red"
    assert row["colour_blurred"] == "blue"
    assert row["prompt_id"] == "neutral"
    assert row["correct_answer"] == "left"
    assert row["parsed_answer"] == "left"
    assert row["correct"] is True


def test_result_to_row_unparseable():
    result = TrialResult(
        condition=Condition(
            crisp_on_top=True,
            crisp_side=Side.left,
            colour_crisp=Colour.red,
            colour_blurred=Colour.blue,
        ),
        model="test-model",
        prompt_id="neutral",
        prompt="test prompt",
        raw_response="gibberish",
        parsed_answer=None,
        correct=None,
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
    )
    row = result_to_row(result)
    assert row["parsed_answer"] is None
    assert row["correct"] is None


def test_append_and_load_roundtrip(tmp_path):
    csv_path = tmp_path / "results.csv"
    results = [_make_result(), _make_result(correct=False)]
    append_results(results, csv_path)

    df = load_results(csv_path)
    assert len(df) == 2
    assert df["model"][0] == "test-model"


def test_append_accumulates(tmp_path):
    csv_path = tmp_path / "results.csv"
    append_results([_make_result()], csv_path)
    append_results([_make_result(model="other-model")], csv_path)

    df = load_results(csv_path)
    assert len(df) == 2
    assert set(df["model"].to_list()) == {"test-model", "other-model"}


def test_incremental_single_row_appends(tmp_path):
    csv_path = tmp_path / "results.csv"
    for i in range(5):
        append_results([_make_result(model=f"model-{i}")], csv_path)

    df = load_results(csv_path)
    assert len(df) == 5
    assert set(df["model"].to_list()) == {f"model-{i}" for i in range(5)}
