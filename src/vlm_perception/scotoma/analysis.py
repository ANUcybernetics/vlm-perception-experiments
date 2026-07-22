"""Analysis for the Scotoma transcription experiment.

Reports the legibility baseline, the bias-index dose-response over blur,
the depth-order effect, and the English vs pseudoword contrast, per
model. Bias index convention: +1 = transcription matches the crisp robot
stream (the exploit works), -1 = matches the blurred real stream (the
model reads like a human), ~0 = mush or both streams recovered.
"""

from pathlib import Path

import polars as pl
from scipy.stats import mannwhitneyu, spearmanr

from vlm_perception.scotoma.storage import load_results


def _fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _mean_bias_table(
    df: pl.DataFrame, group_cols: list[str], metric: str = "bias_index_lev"
) -> pl.DataFrame:
    return (
        df.filter(pl.col(metric).is_not_null())
        .group_by(group_cols)
        .agg(
            pl.col(metric).mean().round(3).alias("mean_bias"),
            pl.col("dist_real_lev").mean().round(3).alias("d_real"),
            pl.col("dist_robot_lev").mean().round(3).alias("d_robot"),
            pl.len().alias("n"),
        )
        .sort(group_cols)
    )


def legibility_report(df: pl.DataFrame) -> str:
    leg = df.filter(pl.col("trial_type") == "legibility")
    if leg.is_empty():
        return "No legibility-baseline trials found.\n"
    table = (
        leg.group_by("model")
        .agg(
            pl.len().alias("n"),
            (pl.col("dist_real_lev") == 0.0).mean().round(3).alias("exact_rate"),
            pl.col("dist_real_lev").mean().round(4).alias("mean_dist_lev"),
            pl.col("dist_real_lev").is_null().sum().alias("unparsed"),
        )
        .sort("model")
    )
    return f"Legibility baseline (solo unblurred strings):\n{table}\n"


def dose_response_report(df: pl.DataFrame, metric: str = "bias_index_lev") -> str:
    """Mean bias vs blur level, exploit conditions (blurred_on_top), per model."""
    main = df.filter(
        (pl.col("trial_type") == "main")
        & pl.col("blurred_on_top")
        & pl.col(metric).is_not_null()
    )
    if main.is_empty():
        return "No main-sweep trials found.\n"
    lines = [f"Dose-response ({metric}, blurred-on-top exploit conditions):"]
    pivot = (
        main.group_by("model", "blur_fraction")
        .agg(pl.col(metric).mean().round(3).alias("bias"), pl.len().alias("n"))
        .pivot(on="blur_fraction", index="model", values="bias")
        .sort("model")
    )
    lines.append(
        str(
            pivot.select(
                "model",
                *sorted((c for c in pivot.columns if c != "model"), key=float),
            )
        )
    )
    lines.append("Spearman trend (bias vs blur fraction), per model:")
    for (model,), sub in sorted(main.group_by("model"), key=lambda kv: kv[0][0]):
        rho, p = spearmanr(sub["blur_fraction"], sub[metric])
        lines.append(f"  {model}: rho={rho:.3f}, p={_fmt_p(p)} (n={len(sub)})")
    return "\n".join(lines) + "\n"


def depth_order_report(df: pl.DataFrame, metric: str = "bias_index_lev") -> str:
    """Exploit (blurred on top) vs congruent control, nonzero blur only."""
    main = df.filter(
        (pl.col("trial_type") == "main")
        & (pl.col("blur_fraction") > 0)
        & pl.col(metric).is_not_null()
    )
    if main.is_empty():
        return "No nonzero-blur main trials found.\n"
    lines = [f"Depth-order effect ({metric}, nonzero blur):"]
    for (model,), sub in sorted(main.group_by("model"), key=lambda kv: kv[0][0]):
        exploit = sub.filter(pl.col("blurred_on_top"))[metric]
        control = sub.filter(~pl.col("blurred_on_top"))[metric]
        if len(exploit) == 0 or len(control) == 0:
            continue
        _, p = mannwhitneyu(exploit, control, alternative="two-sided")
        lines.append(
            f"  {model}: blurred-on-top {exploit.mean():+.3f} (n={len(exploit)}) "
            f"vs crisp-on-top {control.mean():+.3f} (n={len(control)}), "
            f"Mann-Whitney p={_fmt_p(p)}"
        )
    return "\n".join(lines) + "\n"


def pool_contrast_report(df: pl.DataFrame, metric: str = "bias_index_lev") -> str:
    """English vs pseudoword bias, exploit conditions with nonzero blur."""
    main = df.filter(
        (pl.col("trial_type") == "main")
        & (pl.col("blur_fraction") > 0)
        & pl.col("blurred_on_top")
        & pl.col(metric).is_not_null()
    )
    if main.is_empty():
        return "No nonzero-blur exploit trials found.\n"
    lines = [f"English vs pseudoword contrast ({metric}, exploit, nonzero blur):"]
    for (model,), sub in sorted(main.group_by("model"), key=lambda kv: kv[0][0]):
        eng = sub.filter(pl.col("pool") == "english")[metric]
        pse = sub.filter(pl.col("pool") == "pseudo")[metric]
        if len(eng) == 0 or len(pse) == 0:
            continue
        _, p = mannwhitneyu(eng, pse, alternative="two-sided")
        lines.append(
            f"  {model}: english {eng.mean():+.3f} (n={len(eng)}) "
            f"vs pseudo {pse.mean():+.3f} (n={len(pse)}), "
            f"Mann-Whitney p={_fmt_p(p)}"
        )
    return "\n".join(lines) + "\n"


def prompt_report(df: pl.DataFrame, metric: str = "bias_index_lev") -> str:
    main = df.filter(
        (pl.col("trial_type") == "main")
        & (pl.col("blur_fraction") > 0)
        & pl.col("blurred_on_top")
        & pl.col(metric).is_not_null()
    )
    if main.is_empty():
        return ""
    pivot = (
        main.group_by("model", "prompt_id")
        .agg(pl.col(metric).mean().round(3).alias("bias"))
        .pivot(on="prompt_id", index="model", values="bias")
        .sort("model")
    )
    return f"Mean bias by prompt ({metric}, exploit, nonzero blur):\n{pivot}\n"


def full_report(path: Path) -> str:
    df = load_results(path)
    sections = [
        f"Loaded {len(df)} trials from {path}",
        legibility_report(df),
        dose_response_report(df, "bias_index_lev"),
        dose_response_report(df, "bias_index_ham"),
        depth_order_report(df),
        pool_contrast_report(df),
        prompt_report(df),
    ]
    return "\n".join(s for s in sections if s)
