import numpy as np
import polars as pl

from vlm_perception.analysis import (
    _balanced_sweep,
    _cochran_armitage,
    _holm_bonferroni,
    _valid,
    _wilson_ci,
    full_report,
)


def _make_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "model": ["m1"] * 8 + ["m2"] * 4,
            "prompt_id": ["neutral"] * 12,
            "blur_px": [0, 0, 4, 4, 8, 8, 12, 12, 0, 0, 4, 4],
            "crisp_on_top": [True, False] * 6,
            "crisp_side": ["left"] * 12,
            "colour_crisp": ["red"] * 12,
            "colour_blurred": ["cyan"] * 12,
            "correct_answer": ["left"] * 12,
            "parsed_answer": [
                "left",
                "right",
                "left",
                "left",
                "left",
                None,
                "left",
                "right",
            ]
            + ["left"] * 4,
            "correct": [
                True,
                False,
                True,
                True,
                True,
                None,
                True,
                False,
                True,
                True,
                True,
                True,
            ],
        }
    )


# --- helper tests ---


def test_cochran_armitage_strong_trend():
    k = np.array([90, 70, 50, 30, 10])
    n = np.array([100, 100, 100, 100, 100])
    scores = np.array([1, 2, 3, 4, 5])
    z, p = _cochran_armitage(k, n, scores)
    assert z < -10
    assert p < 1e-20


def test_cochran_armitage_no_trend():
    k = np.array([50, 50, 50, 50, 50])
    n = np.array([100, 100, 100, 100, 100])
    scores = np.array([1, 2, 3, 4, 5])
    z, p = _cochran_armitage(k, n, scores)
    assert abs(z) < 0.01
    assert p > 0.9


def test_cochran_armitage_positive_trend():
    k = np.array([10, 30, 50, 70, 90])
    n = np.array([100, 100, 100, 100, 100])
    scores = np.array([1, 2, 3, 4, 5])
    z, p = _cochran_armitage(k, n, scores)
    assert z > 10
    assert p < 1e-20


def test_holm_bonferroni_single():
    assert _holm_bonferroni([0.03]) == [0.03]


def test_holm_bonferroni_correction():
    raw = [0.01, 0.04, 0.03]
    adj = _holm_bonferroni(raw)
    assert adj[0] < 0.05
    assert adj[1] > raw[1]
    assert adj[2] > raw[2]
    assert all(a <= 1.0 for a in adj)


def test_holm_bonferroni_monotonicity():
    raw = [0.001, 0.01, 0.05, 0.10]
    adj = _holm_bonferroni(raw)
    for i in range(len(adj)):
        assert adj[i] >= raw[i]


def test_holm_bonferroni_empty():
    assert _holm_bonferroni([]) == []


def test_wilson_ci_midpoint():
    lo, hi = _wilson_ci(50, 100)
    assert lo < 0.5 < hi
    assert 0.39 < lo < 0.42
    assert 0.58 < hi < 0.61


def test_wilson_ci_extreme():
    lo, hi = _wilson_ci(0, 100)
    assert lo < 1e-10
    assert hi < 0.05


def test_wilson_ci_bounds():
    lo, hi = _wilson_ci(100, 100)
    assert lo > 0.95
    assert hi == 1.0


# --- data filtering tests ---


def test_valid_excludes_nulls():
    df = _make_df()
    valid = _valid(df)
    assert len(valid) == 11
    assert valid["correct"].null_count() == 0


def test_balanced_sweep_filters_colour_pairs():
    df = pl.DataFrame(
        {
            "model": ["m1"] * 4,
            "prompt_id": ["neutral"] * 4,
            "blur_px": [4, 4, 4, 4],
            "crisp_on_top": [True] * 4,
            "crisp_side": ["left"] * 4,
            "colour_crisp": ["red", "red", "blue", "yellow"],
            "colour_blurred": ["cyan", "blue", "green", "blue"],
            "correct": [True] * 4,
        }
    )
    sweep = _balanced_sweep(df)
    assert len(sweep) == 2
    pairs = set(
        zip(
            sweep["colour_crisp"].to_list(),
            sweep["colour_blurred"].to_list(),
            strict=True,
        )
    )
    assert ("red", "cyan") in pairs
    assert ("yellow", "blue") in pairs


# --- report integration test ---


def test_full_report_on_synthetic(tmp_path):
    rows = []
    for model in ["m1", "m2"]:
        for prompt in ["neutral", "minimal"]:
            for blur in [0, 4, 8]:
                for crisp_top in [True, False]:
                    for side in ["left", "right"]:
                        for cc, cb in [("red", "cyan"), ("yellow", "blue")]:
                            correct = crisp_top
                            rows.append(
                                {
                                    "model": model,
                                    "prompt_id": prompt,
                                    "blur_px": blur,
                                    "crisp_on_top": crisp_top,
                                    "crisp_side": side,
                                    "colour_crisp": cc,
                                    "colour_blurred": cb,
                                    "correct_answer": "left",
                                    "parsed_answer": "left" if correct else "right",
                                    "correct": correct,
                                    "prompt": "test",
                                    "raw_response": "test",
                                    "reasoning_trace": None,
                                    "timestamp": "2026-01-01T00:00:00",
                                }
                            )
    df = pl.DataFrame(rows)
    path = tmp_path / "results.jsonl"
    df.write_ndjson(path)
    report = full_report(path)
    assert "Data summary" in report
    assert "crisp-in-front bias" in report
    assert "Accuracy by blur level" in report
    assert "Blur dose-response" in report
    assert "depth order interaction" in report
    assert "Zero-blur baseline" in report
    assert "Model differences" in report
    assert "Prompt effects" in report
    assert "Nuisance variables" in report
    assert "Summary table" in report
    assert "Statistical notes" in report
