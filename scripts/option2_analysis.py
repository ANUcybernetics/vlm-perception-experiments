"""Recompute Table 2 and per-blur ORs under three regimes:

1. all prompts (current paper headline)
2. excluding thinking (proposed option 2 headline)
3. thinking only (separate paragraph in proposed option 2)

Run with:
    uv run python scripts/option2_analysis.py
"""

from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats.contingency import odds_ratio

from vlm_perception.analysis import _balanced_sweep, _valid, _wilson_ci
from vlm_perception.storage import load_results

RESULTS = Path(__file__).resolve().parents[1] / "results" / "results.jsonl"

MODEL_ORDER = [
    "claude-haiku-4-5",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
]


def _2x2(sub_a: pl.DataFrame, sub_b: pl.DataFrame) -> np.ndarray:
    return np.array(
        [
            [int(sub_a["correct"].sum()), int((~sub_a["correct"]).sum())],
            [int(sub_b["correct"].sum()), int((~sub_b["correct"]).sum())],
        ]
    )


def _fmt_or(v: float) -> str:
    if v > 9999:
        return f"{v:.0f}"
    if v >= 100:
        return f"{v:.0f}"
    return f"{v:.1f}"


def per_model_table(df: pl.DataFrame, label: str) -> None:
    print(f"\n## Per-model summary: {label}")
    print(f"{'Model':22s} {'n':>6s} {'Crisp↑':>9s} {'Blurred↑':>10s} "
          f"{'OR':>10s} {'95% CI':>20s} {'0px acc':>10s}")
    print("-" * 92)

    valid = _valid(df)
    for model in [*MODEL_ORDER, "(pooled)"]:
        sub = (
            valid if model == "(pooled)"
            else valid.filter(pl.col("model") == model)
        )
        if len(sub) == 0:
            continue
        ct = sub.filter(pl.col("crisp_on_top"))
        bt = sub.filter(pl.col("crisp_on_top").not_())
        ct_n = len(ct)
        bt_n = len(bt)
        ct_acc = ct["correct"].mean() * 100 if ct_n else 0
        bt_acc = bt["correct"].mean() * 100 if bt_n else 0
        table = _2x2(ct, bt)
        try:
            or_result = odds_ratio(table, kind="conditional")
            or_val = or_result.statistic
            ci = or_result.confidence_interval(confidence_level=0.95)
            or_str = _fmt_or(or_val)
            ci_str = f"[{_fmt_or(ci.low)}, {_fmt_or(ci.high)}]"
        except Exception as exc:  # noqa: BLE001
            or_str = "n/a"
            ci_str = f"({exc})"

        # 0px accuracy (pooled across both depth orders, balanced sweep)
        sweep_sub = _balanced_sweep(sub).filter(pl.col("blur_px") == 0)
        if len(sweep_sub):
            zero_acc = f"{sweep_sub['correct'].mean() * 100:.1f}%"
        else:
            zero_acc = "n/a"

        print(f"{model:22s} {len(sub):>6d} {ct_acc:>8.1f}% {bt_acc:>9.1f}% "
              f"{or_str:>10s} {ci_str:>20s} {zero_acc:>10s}")


def per_blur_pooled(df: pl.DataFrame, label: str) -> None:
    print(f"\n## Per-blur pooled accuracy + OR: {label}")
    sweep = _balanced_sweep(_valid(df))
    print(f"{'Blur':>6s} {'n':>6s} {'Acc(crisp↑)':>13s} "
          f"{'Acc(blur↑)':>12s} {'OR':>10s} {'95% CI':>20s}")
    print("-" * 70)

    for blur in sorted(sweep["blur_px"].unique().to_list()):
        sub = sweep.filter(pl.col("blur_px") == blur)
        ct = sub.filter(pl.col("crisp_on_top"))
        bt = sub.filter(pl.col("crisp_on_top").not_())
        ct_acc = ct["correct"].mean() * 100
        bt_acc = bt["correct"].mean() * 100
        table = _2x2(ct, bt)
        try:
            or_result = odds_ratio(table, kind="conditional")
            ci = or_result.confidence_interval(confidence_level=0.95)
            or_str = _fmt_or(or_result.statistic)
            ci_str = f"[{_fmt_or(ci.low)}, {_fmt_or(ci.high)}]"
        except Exception as exc:  # noqa: BLE001
            or_str = "n/a"
            ci_str = f"({exc})"

        print(f"{blur:>6d} {len(sub):>6d} {ct_acc:>12.1f}% "
              f"{bt_acc:>11.1f}% {or_str:>10s} {ci_str:>20s}")


def main() -> None:
    df = load_results(RESULTS)
    print(f"Total trials loaded: {len(df)}")
    print(f"Prompt variants: {sorted(df['prompt_id'].unique().to_list())}")

    # Regime A: all prompts (current paper)
    per_model_table(df, "ALL PROMPTS (current paper)")
    per_blur_pooled(df, "ALL PROMPTS (current paper)")

    # Regime B: excluding thinking (proposed headline)
    df_no_think = df.filter(pl.col("prompt_id") != "thinking")
    per_model_table(df_no_think, "EXCLUDING THINKING (proposed headline)")
    per_blur_pooled(df_no_think, "EXCLUDING THINKING (proposed headline)")

    # Regime C: thinking only
    df_think = df.filter(pl.col("prompt_id") == "thinking")
    per_model_table(df_think, "THINKING ONLY")
    per_blur_pooled(df_think, "THINKING ONLY")

    # Pooled bias-incongruent accuracy by prompt (for narrative)
    print("\n## Bias-incongruent (blurred-on-top) pooled accuracy, by prompt")
    valid = _valid(df).filter(pl.col("crisp_on_top").not_())
    by_prompt = (
        valid.group_by("prompt_id")
        .agg(
            (pl.col("correct").mean() * 100).alias("acc"),
            pl.col("correct").count().alias("n"),
        )
        .sort("prompt_id")
    )
    print(by_prompt)


if __name__ == "__main__":
    main()
