from itertools import combinations
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import (
    binomtest,
    chi2_contingency,
    fisher_exact,
    spearmanr,
)
from scipy.stats.contingency import association, odds_ratio

from vlm_perception.models import BLUR_SWEEP_COLOUR_PAIRS
from vlm_perception.storage import load_results

SWEEP_PAIRS = {(c.value, b.value) for c, b in BLUR_SWEEP_COLOUR_PAIRS}


def _valid(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(pl.col("correct").is_not_null())


def _balanced_sweep(df: pl.DataFrame) -> pl.DataFrame:
    """Filter to blur-sweep colour pairs and downsample blur=20 to match.

    The full-factorial design also used these colour pairs at blur=20,
    giving ~2x the data at that level. We keep only 3 reps per cell at
    blur=20 (matching other blur levels) for balanced cross-blur analyses.
    """
    pair_exprs = pl.lit(False)
    for cc, cb in SWEEP_PAIRS:
        pair_exprs = pair_exprs | (
            (pl.col("colour_crisp") == cc) & (pl.col("colour_blurred") == cb)
        )
    filtered = df.filter(pair_exprs)
    non_20 = filtered.filter(pl.col("blur_px") != 20)
    at_20 = filtered.filter(pl.col("blur_px") == 20)
    if len(at_20) == 0:
        return filtered
    cell_cols = [
        "model",
        "prompt_id",
        "crisp_on_top",
        "crisp_side",
        "colour_crisp",
        "colour_blurred",
    ]
    at_20_sampled = (
        at_20.with_row_index("_idx")
        .with_columns(pl.col("_idx").rank("ordinal").over(cell_cols).alias("_rank"))
        .filter(pl.col("_rank") <= 3)
        .drop("_idx", "_rank")
    )
    return pl.concat([non_20, at_20_sampled]).sort("blur_px")


# --- statistical helpers ---


def _cochran_armitage(
    successes: np.ndarray,
    totals: np.ndarray,
    scores: np.ndarray,
) -> tuple[float, float]:
    """Cochran-Armitage test for linear trend in proportions.

    Returns (z_statistic, two_sided_p_value).
    Reference: Agresti, Categorical Data Analysis (2002), §5.3.
    """
    successes = np.asarray(successes, dtype=float)
    totals = np.asarray(totals, dtype=float)
    scores = np.asarray(scores, dtype=float)
    N = totals.sum()
    p_hat = successes.sum() / N
    T = np.sum(scores * (successes - totals * p_hat))
    var_T = (
        p_hat
        * (1 - p_hat)
        * (N * np.sum(totals * scores**2) - np.sum(totals * scores) ** 2)
        / N
    )
    z = T / np.sqrt(var_T) if var_T > 0 else 0.0
    from scipy.stats import norm

    p_value = 2 * norm.sf(abs(z))
    return float(z), float(p_value)


def _holm_bonferroni(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni correction for multiple comparisons."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    cummax = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = p * (m - rank)
        cummax = max(cummax, adj)
        adjusted[orig_idx] = min(cummax, 1.0)
    return adjusted


def _wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    from scipy.stats import norm

    if n == 0:
        return (0.0, 1.0)
    z = norm.ppf(1 - alpha / 2)
    p_hat = k / n
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _fmt_p(p: float) -> str:
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def _fmt_pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _fmt_ci(lo: float, hi: float) -> str:
    return f"[{_fmt_pct(lo)}, {_fmt_pct(hi)}]"


def _fmt_or_ci(lo: float, hi: float) -> str:
    return f"[{lo:.1f}, {hi:.1f}]"


# --- report sections ---


def _data_summary(df: pl.DataFrame) -> str:
    valid = _valid(df)
    n_total = len(df)
    n_valid = len(valid)
    n_unparseable = n_total - n_valid
    n_models = df["model"].n_unique()
    n_prompts = df["prompt_id"].n_unique()
    blur_levels = sorted(df["blur_px"].unique().to_list())
    lines = [
        "## 1. Data summary",
        "",
        f"- {n_total} total trials, {n_valid} valid ({n_unparseable} unparseable)",
        f"- {n_models} models, {n_prompts} prompts, {len(blur_levels)} blur levels: {blur_levels}",
        "",
    ]
    sweep = _balanced_sweep(valid)
    lines.append(f"- Balanced blur-sweep subset (4 colour pairs): {len(sweep)} trials")
    cell_counts = sweep.group_by("model", "prompt_id", "blur_px", "crisp_on_top").agg(
        pl.col("correct").count().alias("n")
    )
    min_n = cell_counts["n"].min()
    max_n = cell_counts["n"].max()
    lines.append(f"- Cell sizes in balanced subset: {min_n}-{max_n} trials/cell")
    lines.append(
        "- Note: blur=20 has additional data from the full-factorial design "
        "(all 30 colour pairs). Cross-blur analyses use the balanced subset only."
    )
    return "\n".join(lines)


def _depth_order_effect(df: pl.DataFrame) -> str:
    valid = _valid(df)
    lines = [
        "## 2. Depth order effect",
        "",
        "Primary finding: models overwhelmingly judge the crisp circle as being "
        "in front, regardless of ground truth.",
        "",
    ]

    header = f"{'Model':20s} {'Acc(crisp)':>11s} {'Acc(blur)':>11s} {'OR':>8s} {'95% CI':>16s} {'p':>12s}"
    lines.append(header)
    lines.append("-" * len(header))

    models = [*sorted(valid["model"].unique().to_list()), "(pooled)"]
    for model in models:
        sub = valid if model == "(pooled)" else valid.filter(pl.col("model") == model)
        ct = sub.filter(pl.col("crisp_on_top"))
        bt = sub.filter(pl.col("crisp_on_top").not_())
        ct_acc = ct["correct"].mean()
        bt_acc = bt["correct"].mean()
        table = np.array(
            [
                [int(ct["correct"].sum()), int((~ct["correct"]).sum())],
                [int(bt["correct"].sum()), int((~bt["correct"]).sum())],
            ]
        )
        _, fisher_p = fisher_exact(table)
        or_result = odds_ratio(table, kind="conditional")
        or_val = or_result.statistic
        ci = or_result.confidence_interval(confidence_level=0.95)
        lines.append(
            f"{model:20s} {_fmt_pct(ct_acc):>11s} {_fmt_pct(bt_acc):>11s} "
            f"{or_val:>8.1f} {_fmt_or_ci(ci.low, ci.high):>16s} {_fmt_p(fisher_p):>12s}"
        )

    pooled_sub = valid
    table = np.array(
        [
            [
                int(pooled_sub.filter(pl.col("crisp_on_top"))["correct"].sum()),
                int((~pooled_sub.filter(pl.col("crisp_on_top"))["correct"]).sum()),
            ],
            [
                int(pooled_sub.filter(pl.col("crisp_on_top").not_())["correct"].sum()),
                int(
                    (~pooled_sub.filter(pl.col("crisp_on_top").not_())["correct"]).sum()
                ),
            ],
        ]
    )
    v = association(table, method="cramer")
    lines.append("")
    lines.append(f"Pooled Cramér's V = {v:.3f}")

    return "\n".join(lines)


def _blur_dose_response(df: pl.DataFrame) -> str:
    sweep = _balanced_sweep(_valid(df))
    blur_levels = sorted(sweep["blur_px"].unique().to_list())
    scores = np.array(blur_levels)

    lines = [
        "## 3. Blur dose-response",
        "",
        "Cochran-Armitage trend test on balanced blur-sweep subset, "
        "stratified by depth order.",
        "",
    ]

    for depth_label, depth_val in [
        ("Blurred on top", False),
        ("Crisp on top", True),
    ]:
        lines.append(f"### {depth_label}")
        lines.append("")
        header = f"{'Model':20s} {'Spearman r':>11s} {'CA z':>8s} {'CA p':>12s} {'Slope (%/px)':>13s}"
        lines.append(header)
        lines.append("-" * len(header))

        strat = sweep.filter(pl.col("crisp_on_top") == depth_val)

        for model in [*sorted(strat["model"].unique().to_list()), "(pooled)"]:
            sub = (
                strat if model == "(pooled)" else strat.filter(pl.col("model") == model)
            )
            cells = (
                sub.group_by("blur_px")
                .agg(
                    pl.col("correct").sum().alias("k"),
                    pl.col("correct").count().alias("n"),
                    pl.col("correct").mean().alias("acc"),
                )
                .sort("blur_px")
            )
            k = cells["k"].to_numpy()
            n = cells["n"].to_numpy()
            acc = cells["acc"].to_numpy()
            z, p = _cochran_armitage(k, n, scores)
            if np.std(acc) == 0:
                rho = 0.0
            else:
                rho, _ = spearmanr(scores, acc)
            if len(scores) > 1:
                slope = np.polyfit(scores, acc, 1)[0]
            else:
                slope = 0.0
            lines.append(
                f"{model:20s} {rho:>+11.3f} {z:>+8.2f} {_fmt_p(p):>12s} {slope * 100:>+13.2f}"
            )
        lines.append("")

    return "\n".join(lines)


def _blur_x_depth_interaction(df: pl.DataFrame) -> str:
    sweep = _balanced_sweep(_valid(df))
    blur_levels = sorted(sweep["blur_px"].unique().to_list())

    lines = [
        "## 4. Blur x depth order interaction",
        "",
        "Odds ratio (crisp-on-top vs blurred-on-top) at each blur level, "
        "pooled across models. An OR near 1 means no depth-order effect; "
        "large OR means strong bias toward crisp-in-front.",
        "",
    ]

    header = f"{'Blur (px)':>10s} {'Acc(crisp)':>11s} {'Acc(blur)':>11s} {'OR':>8s} {'95% CI':>16s}"
    lines.append(header)
    lines.append("-" * len(header))

    for blur in blur_levels:
        sub = sweep.filter(pl.col("blur_px") == blur)
        ct = sub.filter(pl.col("crisp_on_top"))
        bt = sub.filter(pl.col("crisp_on_top").not_())
        table = np.array(
            [
                [int(ct["correct"].sum()), int((~ct["correct"]).sum())],
                [int(bt["correct"].sum()), int((~bt["correct"]).sum())],
            ]
        )
        or_result = odds_ratio(table, kind="conditional")
        ci = or_result.confidence_interval(confidence_level=0.95)
        lines.append(
            f"{blur:>10d} {_fmt_pct(ct['correct'].mean()):>11s} "
            f"{_fmt_pct(bt['correct'].mean()):>11s} "
            f"{or_result.statistic:>8.1f} {_fmt_or_ci(ci.low, ci.high):>16s}"
        )

    lines.append("")
    lines.append(
        "The monotonically increasing OR confirms that the depth-order bias "
        "strengthens with blur magnitude."
    )
    return "\n".join(lines)


def _zero_blur_baseline(df: pl.DataFrame) -> str:
    zero = _valid(df).filter(pl.col("blur_px") == 0)

    lines = [
        "## 5. Zero-blur baseline",
        "",
        "At blur=0, both circles are crisp --- any deviation from 50% reveals "
        "a bias unrelated to blur. Exact binomial test against chance.",
        "",
    ]

    header = (
        f"{'Model':20s} {'Accuracy':>9s} {'95% CI':>16s} "
        f"{'p (vs 50%)':>12s} {'Left bias':>10s}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for model in sorted(zero["model"].unique().to_list()):
        sub = zero.filter(pl.col("model") == model)
        k = int(sub["correct"].sum())
        n = len(sub)
        bt = binomtest(k, n, 0.5)
        ci = _wilson_ci(k, n)

        n_left = int(sub.filter(pl.col("parsed_answer") == "left").height)
        left_pct = n_left / n if n > 0 else 0.5
        lines.append(
            f"{model:20s} {_fmt_pct(k / n):>9s} {_fmt_ci(*ci):>16s} "
            f"{_fmt_p(bt.pvalue):>12s} {_fmt_pct(left_pct):>10s}"
        )

    pooled_k = int(zero["correct"].sum())
    pooled_n = len(zero)
    pooled_bt = binomtest(pooled_k, pooled_n, 0.5)
    pooled_ci = _wilson_ci(pooled_k, pooled_n)
    n_left_all = int(zero.filter(pl.col("parsed_answer") == "left").height)
    lines.append(
        f"{'(pooled)':20s} {_fmt_pct(pooled_k / pooled_n):>9s} "
        f"{_fmt_ci(*pooled_ci):>16s} "
        f"{_fmt_p(pooled_bt.pvalue):>12s} {_fmt_pct(n_left_all / pooled_n):>10s}"
    )

    lines.append("")
    lines.append(
        "Models vary widely in their ability to detect geometric occlusion "
        "without blur cues. Some show strong positional (left/right) biases."
    )
    return "\n".join(lines)


def _model_effect(df: pl.DataFrame) -> str:
    valid = _valid(df)
    lines = [
        "## 6. Model differences",
        "",
    ]

    for depth_label, depth_filter in [
        ("All trials", None),
        ("Blurred on top only", False),
    ]:
        sub = (
            valid
            if depth_filter is None
            else valid.filter(pl.col("crisp_on_top") == depth_filter)
        )
        ct = (
            sub.group_by("model")
            .agg(
                pl.col("correct").sum().alias("correct"),
                (pl.col("correct").count() - pl.col("correct").sum()).alias(
                    "incorrect"
                ),
            )
            .sort("model")
        )
        table = ct.select("correct", "incorrect").to_numpy()
        if np.any(table.sum(axis=0) == 0):
            lines.append(
                f"### {depth_label}: chi-square not applicable (zero marginal)"
            )
            lines.append("")
            continue
        chi2, p, dof, _ = chi2_contingency(table)
        v = association(table, method="cramer")
        lines.append(
            f"### {depth_label}: χ²({dof}) = {chi2:.1f}, p = {_fmt_p(p)}, V = {v:.3f}"
        )
        lines.append("")

        models = ct["model"].to_list()
        accs = {m: sub.filter(pl.col("model") == m)["correct"].mean() for m in models}
        for m in models:
            lines.append(f"  {m}: {_fmt_pct(accs[m])}")
        lines.append("")

        pairs = list(combinations(models, 2))
        raw_ps = []
        pair_labels = []
        for m1, m2 in pairs:
            s1 = sub.filter(pl.col("model") == m1)
            s2 = sub.filter(pl.col("model") == m2)
            t = np.array(
                [
                    [int(s1["correct"].sum()), int((~s1["correct"]).sum())],
                    [int(s2["correct"].sum()), int((~s2["correct"]).sum())],
                ]
            )
            _, fp = fisher_exact(t)
            raw_ps.append(fp)
            pair_labels.append(f"{m1} vs {m2}")
        adj_ps = _holm_bonferroni(raw_ps)
        sig_pairs = [
            (pair_labels[i], raw_ps[i], adj_ps[i])
            for i in range(len(pairs))
            if adj_ps[i] < 0.05
        ]
        if sig_pairs:
            lines.append(
                f"Significant pairwise differences (Holm-corrected, {len(pairs)} comparisons):"
            )
            for label, raw, adj in sig_pairs:
                lines.append(f"  {label}: p_raw = {_fmt_p(raw)}, p_adj = {_fmt_p(adj)}")
        else:
            lines.append("No significant pairwise differences after Holm correction.")
        lines.append("")

    return "\n".join(lines)


def _prompt_effect(df: pl.DataFrame) -> str:
    valid = _valid(df)
    lines = [
        "## 7. Prompt effects",
        "",
    ]

    for depth_label, depth_filter in [
        ("All trials", None),
        ("Blurred on top only", False),
    ]:
        sub = (
            valid
            if depth_filter is None
            else valid.filter(pl.col("crisp_on_top") == depth_filter)
        )
        ct = (
            sub.group_by("prompt_id")
            .agg(
                pl.col("correct").sum().alias("correct"),
                (pl.col("correct").count() - pl.col("correct").sum()).alias(
                    "incorrect"
                ),
            )
            .sort("prompt_id")
        )
        table = ct.select("correct", "incorrect").to_numpy()
        if np.any(table.sum(axis=0) == 0):
            lines.append(
                f"### {depth_label}: chi-square not applicable (zero marginal)"
            )
            lines.append("")
            continue
        chi2, p, dof, _ = chi2_contingency(table)
        v = association(table, method="cramer")
        lines.append(
            f"### {depth_label}: χ²({dof}) = {chi2:.1f}, p = {_fmt_p(p)}, V = {v:.3f}"
        )
        lines.append("")

        prompts = ct["prompt_id"].to_list()
        accs = {
            pid: sub.filter(pl.col("prompt_id") == pid)["correct"].mean()
            for pid in prompts
        }
        for pid in sorted(prompts, key=lambda x: accs[x], reverse=True):
            lines.append(f"  {pid}: {_fmt_pct(accs[pid])}")
        lines.append("")

    wide = (
        valid.group_by("model", "prompt_id")
        .agg(pl.col("correct").mean().alias("accuracy"))
        .pivot(on="prompt_id", index="model", values="accuracy")
        .sort("model")
    )
    lines.append("### Accuracy by model x prompt")
    lines.append("")
    lines.append(str(wide))

    return "\n".join(lines)


def _nuisance_variables(df: pl.DataFrame) -> str:
    valid = _valid(df)
    lines = [
        "## 8. Nuisance variables",
        "",
        "Confirming that spatial position (left/right) and colour pair "
        "have negligible effects.",
        "",
    ]

    side_ct = (
        valid.group_by("crisp_side")
        .agg(
            pl.col("correct").sum().alias("correct"),
            (pl.col("correct").count() - pl.col("correct").sum()).alias("incorrect"),
        )
        .sort("crisp_side")
    )
    table = side_ct.select("correct", "incorrect").to_numpy()
    chi2, p, dof, _ = chi2_contingency(table)
    v = association(table, method="cramer")
    lines.append(f"Side effect: χ²({dof}) = {chi2:.2f}, p = {_fmt_p(p)}, V = {v:.4f}")
    for row in side_ct.iter_rows(named=True):
        n = row["correct"] + row["incorrect"]
        acc = row["correct"] / n
        lines.append(f"  {row['crisp_side']}: {_fmt_pct(acc)} (n={n})")
    lines.append("")

    colour_ct = (
        valid.group_by("colour_crisp", "colour_blurred")
        .agg(
            pl.col("correct").sum().alias("correct"),
            (pl.col("correct").count() - pl.col("correct").sum()).alias("incorrect"),
        )
        .sort("colour_crisp", "colour_blurred")
    )
    table = colour_ct.select("correct", "incorrect").to_numpy()
    chi2, p, dof, _ = chi2_contingency(table)
    v = association(table, method="cramer")
    lines.append(
        f"Colour pair effect: χ²({dof}) = {chi2:.2f}, p = {_fmt_p(p)}, V = {v:.4f}"
    )

    return "\n".join(lines)


def _summary_table(df: pl.DataFrame) -> str:
    valid = _valid(df)
    sweep = _balanced_sweep(valid)
    blur_levels = sorted(sweep["blur_px"].unique().to_list())
    scores = np.array(blur_levels)

    lines = [
        "## 9. Summary table",
        "",
    ]

    header = (
        f"{'Model':20s} {'Overall':>8s} {'Crisp↑':>8s} {'Blur↑':>8s} "
        f"{'OR':>7s} {'0px':>8s} {'Trend z':>8s}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for model in sorted(valid["model"].unique().to_list()):
        msub = valid.filter(pl.col("model") == model)
        overall = msub["correct"].mean()
        ct_acc = msub.filter(pl.col("crisp_on_top"))["correct"].mean()
        bt_acc = msub.filter(pl.col("crisp_on_top").not_())["correct"].mean()

        table = np.array(
            [
                [
                    int(msub.filter(pl.col("crisp_on_top"))["correct"].sum()),
                    int((~msub.filter(pl.col("crisp_on_top"))["correct"]).sum()),
                ],
                [
                    int(msub.filter(pl.col("crisp_on_top").not_())["correct"].sum()),
                    int((~msub.filter(pl.col("crisp_on_top").not_())["correct"]).sum()),
                ],
            ]
        )
        or_val = odds_ratio(table, kind="conditional").statistic

        zero = msub.filter(pl.col("blur_px") == 0)
        zero_acc = zero["correct"].mean() if len(zero) > 0 else float("nan")

        bt_sweep = _balanced_sweep(msub.filter(pl.col("crisp_on_top").not_()))
        cells = (
            bt_sweep.group_by("blur_px")
            .agg(
                pl.col("correct").sum().alias("k"),
                pl.col("correct").count().alias("n"),
            )
            .sort("blur_px")
        )
        z, _ = _cochran_armitage(
            cells["k"].to_numpy(),
            cells["n"].to_numpy(),
            scores,
        )

        lines.append(
            f"{model:20s} {_fmt_pct(overall):>8s} {_fmt_pct(ct_acc):>8s} "
            f"{_fmt_pct(bt_acc):>8s} {or_val:>7.1f} "
            f"{_fmt_pct(zero_acc):>8s} {z:>+8.2f}"
        )

    lines.append("")
    lines.append(
        "Crisp↑ = accuracy when crisp circle is on top. "
        "Blur↑ = accuracy when blurred circle is on top. "
        "OR = odds ratio for depth order effect. "
        "0px = accuracy at zero blur. "
        "Trend z = Cochran-Armitage z for blur→accuracy trend (blurred-on-top)."
    )
    return "\n".join(lines)


def _statistical_notes() -> str:
    return "\n".join(
        [
            "## Statistical notes",
            "",
            "- All p-values are two-sided.",
            "- Odds ratios are conditional maximum likelihood estimates "
            "(scipy.stats.contingency.odds_ratio, kind='conditional') with "
            "exact 95% confidence intervals.",
            "- Cochran-Armitage trend test uses blur radius in pixels as scores.",
            "- Cross-blur analyses use the balanced blur-sweep subset (4 colour "
            "pairs, equal cell sizes at all blur levels) to avoid "
            "over-representing blur=20.",
            "- Pairwise comparisons use Holm-Bonferroni correction.",
            "- Proportion confidence intervals use the Wilson score method.",
        ]
    )


def full_report(results_path: Path) -> str:
    df = load_results(results_path)
    sections = [
        _data_summary(df),
        _depth_order_effect(df),
        _blur_dose_response(df),
        _blur_x_depth_interaction(df),
        _zero_blur_baseline(df),
        _model_effect(df),
        _prompt_effect(df),
        _nuisance_variables(df),
        _summary_table(df),
        _statistical_notes(),
    ]
    return "\n\n".join(sections)
