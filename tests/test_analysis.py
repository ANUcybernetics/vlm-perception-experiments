import polars as pl

from vlm_perception.analysis import (
    accuracy_by_layout,
    accuracy_by_side,
    depth_order_fisher_test,
    overall_accuracy,
    unparseable_count,
)


def _make_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "model": ["m1", "m1", "m1", "m1", "m2", "m2"],
            "crisp_on_top": [True, True, False, False, True, False],
            "crisp_side": ["left", "left", "right", "right", "left", "right"],
            "colour_crisp": ["red", "blue", "green", "red", "red", "red"],
            "colour_blurred": ["blue", "green", "red", "blue", "blue", "blue"],
            "correct_answer": ["left", "left", "left", "left", "left", "left"],
            "parsed_answer": ["left", "right", "left", None, "left", "right"],
            "correct": [True, False, True, None, True, False],
        }
    )


def test_overall_accuracy():
    result = overall_accuracy(_make_df())
    assert len(result) == 2

    m1 = result.filter(pl.col("model") == "m1")
    assert m1["n_correct"][0] == 2
    assert m1["n_total"][0] == 3

    m2 = result.filter(pl.col("model") == "m2")
    assert m2["n_correct"][0] == 1
    assert m2["n_total"][0] == 2


def test_accuracy_by_layout():
    result = accuracy_by_layout(_make_df())
    m1_crisp = result.filter((pl.col("model") == "m1") & pl.col("crisp_on_top"))
    assert m1_crisp["n_correct"][0] == 1
    assert m1_crisp["n_total"][0] == 2


def test_accuracy_by_side():
    result = accuracy_by_side(_make_df())
    assert len(result) > 0
    assert "crisp_side" in result.columns


def test_unparseable_count():
    result = unparseable_count(_make_df())
    assert len(result) == 1
    assert result["n_unparseable"][0] == 1


def _make_fisher_df() -> pl.DataFrame:
    """Model that always picks the crisp circle: 100% when crisp on top, 0% when blurred on top."""
    rows = []
    for crisp_on_top in [True, False]:
        for _ in range(10):
            correct = crisp_on_top
            rows.append(
                {
                    "model": "biased",
                    "crisp_on_top": crisp_on_top,
                    "crisp_side": "left",
                    "colour_crisp": "red",
                    "colour_blurred": "blue",
                    "correct_answer": "left",
                    "parsed_answer": "left",
                    "correct": correct,
                }
            )
    return pl.DataFrame(rows)


def test_depth_order_fisher_test_significant():
    result = depth_order_fisher_test(_make_fisher_df())
    assert len(result) == 1
    assert result["model"][0] == "biased"
    assert result["p_value"][0] < 0.001


def test_depth_order_fisher_test_not_significant():
    df = pl.DataFrame(
        {
            "model": ["fair"] * 20,
            "crisp_on_top": [True] * 10 + [False] * 10,
            "crisp_side": ["left"] * 20,
            "colour_crisp": ["red"] * 20,
            "colour_blurred": ["blue"] * 20,
            "correct_answer": ["left"] * 20,
            "parsed_answer": ["left"] * 20,
            "correct": [True, False] * 10,
        }
    )
    result = depth_order_fisher_test(df)
    assert result["p_value"][0] > 0.05
