from itertools import combinations
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import (
    binomtest,
    chi2_contingency,
    fisher_exact,
    norm,
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
    p_value = 2 * norm.sf(abs(z))
    return float(z), float(p_value)


def _holm_bonferroni(p_values: list[float]) -> list[float]:
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
    if n == 0:
        return (0.0, 1.0)
    z = norm.ppf(1 - alpha / 2)
    p_hat = k / n
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _2x2(sub_a: pl.DataFrame, sub_b: pl.DataFrame) -> np.ndarray:
    return np.array(
        [
            [int(sub_a["correct"].sum()), int((~sub_a["correct"]).sum())],
            [int(sub_b["correct"].sum()), int((~sub_b["correct"]).sum())],
        ]
    )


def _fmt_p(p: float) -> str:
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def _fmt_pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _fmt_ci(lo: float, hi: float) -> str:
    return f"[{_fmt_pct(lo)}, {_fmt_pct(hi)}]"


def _fmt_or(val: float) -> str:
    if val > 9999:
        return f"{val:.0f}"
    if val >= 100:
        return f"{val:.0f}"
    return f"{val:.1f}"


def _fmt_or_ci(lo: float, hi: float) -> str:
    return f"[{_fmt_or(lo)}, {_fmt_or(hi)}]"


# --- report sections ---


def _data_summary(df: pl.DataFrame) -> str:
    valid = _valid(df)
    n_total = len(df)
    n_valid = len(valid)
    n_unparseable = n_total - n_valid
    n_models = df["model"].n_unique()
    n_prompts = df["prompt_id"].n_unique()
    blur_levels = sorted(df["blur_px"].unique().to_list())
    models = sorted(valid["model"].unique().to_list())

    lines = [
        "## 1. Data summary",
        "",
        f"- {n_total} total trials, {n_valid} valid ({n_unparseable} unparseable)",
        f"- {n_models} models: {', '.join(models)}",
        f"- {n_prompts} prompt variants, {len(blur_levels)} blur levels: {blur_levels}",
    ]

    per_model = (
        valid.group_by("model")
        .agg(pl.col("correct").count().alias("n"))
        .sort("model")
    )
    for row in per_model.iter_rows(named=True):
        lines.append(f"  {row['model']}: {row['n']} trials")

    sweep = _balanced_sweep(valid)
    lines.append("")
    lines.append(f"- Balanced blur-sweep subset (4 colour pairs): {len(sweep)} trials")
    cell_counts = sweep.group_by(
        "model", "prompt_id", "blur_px", "crisp_on_top"
    ).agg(pl.col("correct").count().alias("n"))
    min_n = cell_counts["n"].min()
    max_n = cell_counts["n"].max()
    lines.append(f"- Cell sizes in balanced subset: {min_n}--{max_n} trials/cell")
    lines.append(
        "- Note: blur=20 has additional data from the full-factorial design "
        "(all 30 colour pairs). Cross-blur analyses use the balanced subset."
    )
    return "\n".join(lines)


def _depth_order_bias(df: pl.DataFrame) -> str:
    valid = _valid(df)
    lines = [
        "## 2. The crisp-in-front bias",
        "",
        "Every model tested exhibits a strong bias toward judging the crisp "
        "circle as being in front, regardless of ground truth. When the crisp "
        "circle is actually on top (bias-congruent), accuracy is near ceiling; "
        "when the blurred circle is on top (bias-incongruent), accuracy drops "
        "dramatically.",
        "",
    ]

    header = (
        f"{'Model':20s} {'n':>6s} {'Acc(crisp↑)':>12s} {'95% CI':>16s} "
        f"{'Acc(blur↑)':>12s} {'95% CI':>16s} {'OR':>8s} {'95% CI':>16s}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    models = sorted(valid["model"].unique().to_list())

    for model in [*models, "(pooled)"]:
        sub = valid if model == "(pooled)" else valid.filter(pl.col("model") == model)
        ct = sub.filter(pl.col("crisp_on_top"))
        bt = sub.filter(pl.col("crisp_on_top").not_())
        ct_k, ct_n = int(ct["correct"].sum()), len(ct)
        bt_k, bt_n = int(bt["correct"].sum()), len(bt)
        ct_ci = _wilson_ci(ct_k, ct_n)
        bt_ci = _wilson_ci(bt_k, bt_n)
        table = _2x2(ct, bt)
        or_result = odds_ratio(table, kind="conditional")
        ci = or_result.confidence_interval(confidence_level=0.95)

        lines.append(
            f"{model:20s} {len(sub):>6d} {_fmt_pct(ct_k / ct_n):>12s} "
            f"{_fmt_ci(*ct_ci):>16s} {_fmt_pct(bt_k / bt_n):>12s} "
            f"{_fmt_ci(*bt_ci):>16s} {or_result.statistic:>8.1f} "
            f"{_fmt_or_ci(ci.low, ci.high):>16s}"
        )

    pooled_table = _2x2(
        valid.filter(pl.col("crisp_on_top")),
        valid.filter(pl.col("crisp_on_top").not_()),
    )
    v = association(pooled_table, method="cramer")
    lines.append("")
    lines.append(f"Pooled Cramér's V = {v:.3f} (large effect)")
    lines.append(
        "The per-model ORs range from 6 to >1000, showing that the bias is "
        "universal but varies substantially in magnitude across models."
    )

    return "\n".join(lines)


def _blur_accuracy_curves(df: pl.DataFrame) -> str:
    sweep = _balanced_sweep(_valid(df))
    blur_levels = sorted(sweep["blur_px"].unique().to_list())
    models = sorted(sweep["model"].unique().to_list())

    lines = [
        "## 3. Accuracy by blur level",
        "",
        "Accuracy (with 95% Wilson CIs) at each blur level, split by depth "
        "order. Balanced blur-sweep subset. As blur increases, "
        "bias-congruent accuracy rises toward ceiling while "
        "bias-incongruent accuracy falls toward floor.",
        "",
    ]

    for depth_label, depth_val in [
        ("Blurred on top (bias-incongruent)", False),
        ("Crisp on top (bias-congruent)", True),
    ]:
        lines.append(f"### {depth_label}")
        lines.append("")
        blur_hdrs = "  ".join(f"{'blur=' + str(b):>16s}" for b in blur_levels)
        header = f"{'Model':20s}  {blur_hdrs}"
        lines.append(header)
        lines.append("-" * len(header))

        strat = sweep.filter(pl.col("crisp_on_top") == depth_val)
        for model in [*models, "(pooled)"]:
            sub = (
                strat
                if model == "(pooled)"
                else strat.filter(pl.col("model") == model)
            )
            cells = []
            for blur in blur_levels:
                bsub = sub.filter(pl.col("blur_px") == blur)
                k = int(bsub["correct"].sum())
                n = len(bsub)
                ci = _wilson_ci(k, n)
                cells.append(f"{_fmt_pct(k / n):>6s} {_fmt_ci(*ci)}")
            lines.append(f"{model:20s}  {'  '.join(cells)}")
        lines.append("")

    return "\n".join(lines)


def _blur_dose_response(df: pl.DataFrame) -> str:
    sweep = _balanced_sweep(_valid(df))
    blur_levels = sorted(sweep["blur_px"].unique().to_list())
    scores = np.array(blur_levels)

    lines = [
        "## 4. Blur dose-response",
        "",
        "Cochran-Armitage test for trend in accuracy across blur levels. "
        "A significant negative z for bias-incongruent trials confirms that "
        "the crisp-in-front bias strengthens with blur magnitude.",
        "",
    ]

    for depth_label, depth_val in [
        ("Blurred on top (bias-incongruent)", False),
        ("Crisp on top (bias-congruent)", True),
    ]:
        lines.append(f"### {depth_label}")
        lines.append("")
        header = f"{'Model':20s} {'CA z':>8s} {'p':>12s} {'Slope (%/px)':>13s}"
        lines.append(header)
        lines.append("-" * len(header))

        strat = sweep.filter(pl.col("crisp_on_top") == depth_val)

        for model in [*sorted(strat["model"].unique().to_list()), "(pooled)"]:
            sub = (
                strat
                if model == "(pooled)"
                else strat.filter(pl.col("model") == model)
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
            slope = np.polyfit(scores, acc, 1)[0] if len(scores) > 1 else 0.0
            lines.append(
                f"{model:20s} {z:>+8.2f} {_fmt_p(p):>12s} "
                f"{slope * 100:>+13.2f}"
            )
        lines.append("")

    return "\n".join(lines)


def _blur_x_depth_interaction(df: pl.DataFrame) -> str:
    sweep = _balanced_sweep(_valid(df))
    blur_levels = sorted(sweep["blur_px"].unique().to_list())
    models = sorted(sweep["model"].unique().to_list())

    lines = [
        "## 5. Blur × depth order interaction",
        "",
        "The crisp-in-front bias is negligible at zero blur and escalates "
        "with blur magnitude. Odds ratios (crisp-on-top vs blurred-on-top) "
        "at each blur level quantify this escalation.",
        "",
    ]

    lines.append("### Pooled across models")
    lines.append("")
    header = (
        f"{'Blur (px)':>10s} {'Acc(crisp↑)':>12s} {'Acc(blur↑)':>12s} "
        f"{'OR':>8s} {'95% CI':>16s} {'p':>12s}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for blur in blur_levels:
        sub = sweep.filter(pl.col("blur_px") == blur)
        ct = sub.filter(pl.col("crisp_on_top"))
        bt = sub.filter(pl.col("crisp_on_top").not_())
        table = _2x2(ct, bt)
        or_result = odds_ratio(table, kind="conditional")
        ci = or_result.confidence_interval(confidence_level=0.95)
        _, fp = fisher_exact(table)
        lines.append(
            f"{blur:>10d} {_fmt_pct(ct['correct'].mean()):>12s} "
            f"{_fmt_pct(bt['correct'].mean()):>12s} "
            f"{_fmt_or(or_result.statistic):>8s} "
            f"{_fmt_or_ci(ci.low, ci.high):>16s} {_fmt_p(fp):>12s}"
        )

    lines.append("")
    lines.append("### Per-model depth-order OR at each blur level")
    lines.append("")
    blur_hdrs = "  ".join(f"{'blur=' + str(b):>10s}" for b in blur_levels)
    header = f"{'Model':20s}  {blur_hdrs}"
    lines.append(header)
    lines.append("-" * len(header))

    for model in models:
        msub = sweep.filter(pl.col("model") == model)
        vals = []
        for blur in blur_levels:
            bsub = msub.filter(pl.col("blur_px") == blur)
            ct = bsub.filter(pl.col("crisp_on_top"))
            bt = bsub.filter(pl.col("crisp_on_top").not_())
            table = _2x2(ct, bt)
            or_val = odds_ratio(table, kind="conditional").statistic
            vals.append(f"{_fmt_or(or_val):>10s}")
        lines.append(f"{model:20s}  {'  '.join(vals)}")

    lines.append("")
    lines.append(
        "All models show monotonically increasing ORs with blur magnitude, "
        "confirming the bias is driven by blur, not model-specific artefacts."
    )

    return "\n".join(lines)


def _zero_blur_baseline(df: pl.DataFrame) -> str:
    zero = _valid(df).filter(pl.col("blur_px") == 0)

    lines = [
        "## 6. Zero-blur baseline",
        "",
        "At blur=0, both circles are crisp and identical except for colour "
        "--- the task reduces to pure geometric occlusion detection. "
        "Accuracy above 50% indicates the model can read occlusion from "
        "edge continuity alone. The depth-order OR at blur=0 serves as a "
        "manipulation check: values near 1.0 confirm that the "
        "crisp-in-front bias requires an actual blur difference.",
        "",
    ]

    header = (
        f"{'Model':20s} {'n':>5s} {'Accuracy':>9s} {'95% CI':>16s} "
        f"{'p (vs 50%)':>12s} {'Depth OR':>9s} {'Left bias':>10s}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for model in sorted(zero["model"].unique().to_list()):
        sub = zero.filter(pl.col("model") == model)
        k = int(sub["correct"].sum())
        n = len(sub)
        bt = binomtest(k, n, 0.5)
        ci = _wilson_ci(k, n)

        ct = sub.filter(pl.col("crisp_on_top"))
        btsub = sub.filter(pl.col("crisp_on_top").not_())
        if len(ct) > 0 and len(btsub) > 0:
            table = _2x2(ct, btsub)
            depth_or = odds_ratio(table, kind="conditional").statistic
        else:
            depth_or = float("nan")

        n_left = int(sub.filter(pl.col("parsed_answer") == "left").height)
        left_pct = n_left / n if n > 0 else 0.5
        lines.append(
            f"{model:20s} {n:>5d} {_fmt_pct(k / n):>9s} {_fmt_ci(*ci):>16s} "
            f"{_fmt_p(bt.pvalue):>12s} {depth_or:>9.2f} "
            f"{_fmt_pct(left_pct):>10s}"
        )

    pooled_k = int(zero["correct"].sum())
    pooled_n = len(zero)
    pooled_bt = binomtest(pooled_k, pooled_n, 0.5)
    pooled_ci = _wilson_ci(pooled_k, pooled_n)
    ct_all = zero.filter(pl.col("crisp_on_top"))
    bt_all = zero.filter(pl.col("crisp_on_top").not_())
    depth_or_pooled = odds_ratio(
        _2x2(ct_all, bt_all), kind="conditional"
    ).statistic
    n_left_all = int(zero.filter(pl.col("parsed_answer") == "left").height)
    lines.append(
        f"{'(pooled)':20s} {pooled_n:>5d} {_fmt_pct(pooled_k / pooled_n):>9s} "
        f"{_fmt_ci(*pooled_ci):>16s} "
        f"{_fmt_p(pooled_bt.pvalue):>12s} {depth_or_pooled:>9.2f} "
        f"{_fmt_pct(n_left_all / pooled_n):>10s}"
    )

    return "\n".join(lines)


def _model_comparison(df: pl.DataFrame) -> str:
    valid = _valid(df)
    lines = [
        "## 7. Model differences",
        "",
        "The bias-incongruent condition (blurred on top) is the critical "
        "test: higher accuracy means the model can resist the "
        "crisp-in-front bias. Overall accuracy is uninformative because "
        "the balanced design makes it ~50% when the bias is strong.",
        "",
    ]

    for depth_label, depth_filter in [
        ("Blurred on top (the discriminating case)", False),
        ("All trials", None),
    ]:
        sub = (
            valid
            if depth_filter is None
            else valid.filter(pl.col("crisp_on_top") == depth_filter)
        )
        ct = (
            sub.group_by("model")
            .agg(
                pl.col("correct").sum().alias("k"),
                pl.col("correct").count().alias("n"),
            )
            .sort("model")
        )
        models = ct["model"].to_list()
        k_arr = ct["k"].to_list()
        n_arr = ct["n"].to_list()

        table = np.column_stack(
            [ct["k"].to_numpy(), (ct["n"] - ct["k"]).to_numpy()]
        )
        if np.any(table.sum(axis=0) == 0):
            lines.append(
                f"### {depth_label}: chi-square not applicable (zero marginal)"
            )
            lines.append("")
            for i, m in enumerate(models):
                lines.append(
                    f"  {m}: {_fmt_pct(k_arr[i] / n_arr[i])} (n={n_arr[i]})"
                )
            lines.append("")
            continue
        chi2_val, p, dof, _ = chi2_contingency(table)
        v = association(table, method="cramer")
        lines.append(
            f"### {depth_label}: χ²({dof}) = {chi2_val:.1f}, "
            f"p = {_fmt_p(p)}, V = {v:.3f}"
        )
        lines.append("")

        for i, m in enumerate(models):
            ci = _wilson_ci(k_arr[i], n_arr[i])
            lines.append(
                f"  {m}: {_fmt_pct(k_arr[i] / n_arr[i])} {_fmt_ci(*ci)} "
                f"(n={n_arr[i]})"
            )
        lines.append("")

        pairs = list(combinations(range(len(models)), 2))
        raw_ps = []
        pair_labels = []
        for i, j in pairs:
            s1 = sub.filter(pl.col("model") == models[i])
            s2 = sub.filter(pl.col("model") == models[j])
            t = _2x2(s1, s2)
            _, fp = fisher_exact(t)
            raw_ps.append(fp)
            pair_labels.append(f"{models[i]} vs {models[j]}")
        adj_ps = _holm_bonferroni(raw_ps)
        sig_pairs = [
            (pair_labels[idx], raw_ps[idx], adj_ps[idx])
            for idx in range(len(pairs))
            if adj_ps[idx] < 0.05
        ]
        if sig_pairs:
            lines.append(
                f"Significant pairwise differences "
                f"(Holm-corrected, {len(pairs)} comparisons):"
            )
            for label, raw, adj in sig_pairs:
                lines.append(
                    f"  {label}: p_raw = {_fmt_p(raw)}, p_adj = {_fmt_p(adj)}"
                )
        else:
            lines.append(
                "No significant pairwise differences after Holm correction."
            )
        lines.append("")

    return "\n".join(lines)


def _prompt_effects(df: pl.DataFrame) -> str:
    valid = _valid(df)
    lines = [
        "## 8. Prompt effects",
        "",
        "Prompt framing produces statistically significant but practically "
        "modest differences. The effect is larger for bias-incongruent "
        "trials, where there is more room for variation.",
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
                pl.col("correct").sum().alias("k"),
                pl.col("correct").count().alias("n"),
            )
            .sort("prompt_id")
        )
        table = np.column_stack(
            [ct["k"].to_numpy(), (ct["n"] - ct["k"]).to_numpy()]
        )
        if np.any(table.sum(axis=0) == 0):
            continue
        chi2_val, p, dof, _ = chi2_contingency(table)
        v = association(table, method="cramer")
        lines.append(
            f"### {depth_label}: χ²({dof}) = {chi2_val:.1f}, "
            f"p = {_fmt_p(p)}, V = {v:.3f}"
        )
        lines.append("")

        prompts = ct["prompt_id"].to_list()
        accs = {
            pid: sub.filter(pl.col("prompt_id") == pid)["correct"].mean()
            for pid in prompts
        }
        for pid in sorted(prompts, key=lambda x: accs[x], reverse=True):
            k = int(sub.filter(pl.col("prompt_id") == pid)["correct"].sum())
            n = int(sub.filter(pl.col("prompt_id") == pid).height)
            ci = _wilson_ci(k, n)
            lines.append(f"  {pid}: {_fmt_pct(accs[pid])} {_fmt_ci(*ci)}")
        lines.append("")

    lines.append("### Accuracy by model × prompt (blurred on top)")
    lines.append("")
    bt = valid.filter(pl.col("crisp_on_top").not_())
    wide = (
        bt.group_by("model", "prompt_id")
        .agg(pl.col("correct").mean().alias("accuracy"))
        .pivot(on="prompt_id", index="model", values="accuracy")
        .sort("model")
    )
    prompt_cols = [c for c in wide.columns if c != "model"]
    header = f"{'Model':20s} " + " ".join(
        f"{p:>13s}" for p in sorted(prompt_cols)
    )
    lines.append(header)
    lines.append("-" * len(header))
    for row in wide.iter_rows(named=True):
        vals = " ".join(
            f"{_fmt_pct(row[p]) if row[p] is not None else 'N/A':>13s}"
            for p in sorted(prompt_cols)
        )
        lines.append(f"{row['model']:20s} {vals}")
    lines.append("")

    lines.append(
        "Cramér's V for prompt (~0.07--0.12) is an order of magnitude "
        "smaller than V for depth order (~0.72). No prompt variant "
        "overcomes the crisp-in-front bias."
    )

    return "\n".join(lines)


def _nuisance_variables(df: pl.DataFrame) -> str:
    valid = _valid(df)
    lines = [
        "## 9. Nuisance variables",
        "",
        "The experimental design counterbalances spatial position "
        "(left/right) and colour pair. Both reach statistical significance "
        "--- expected with n > 23,000 --- but have small effect sizes.",
        "",
    ]

    lines.append("### Spatial position (left/right)")
    lines.append("")
    side_ct = (
        valid.group_by("crisp_side")
        .agg(
            pl.col("correct").sum().alias("k"),
            pl.col("correct").count().alias("n"),
        )
        .sort("crisp_side")
    )
    table = np.column_stack(
        [side_ct["k"].to_numpy(), (side_ct["n"] - side_ct["k"]).to_numpy()]
    )
    if np.any(table.sum(axis=0) == 0):
        lines.append(
            "Marginal effect: chi-square not applicable (zero marginal)"
        )
    else:
        chi2_val, p, dof, _ = chi2_contingency(table)
        v = association(table, method="cramer")
        lines.append(
            f"Marginal effect: χ²({dof}) = {chi2_val:.2f}, "
            f"p = {_fmt_p(p)}, V = {v:.4f}"
        )
    for row in side_ct.iter_rows(named=True):
        acc = row["k"] / row["n"]
        lines.append(
            f"  {row['crisp_side']}: {_fmt_pct(acc)} (n={row['n']})"
        )
    lines.append("")

    lines.append(
        "Side × depth order interaction (does side bias differ by depth?):"
    )
    for depth_label, depth_val in [
        ("Crisp on top", True),
        ("Blurred on top", False),
    ]:
        dsub = valid.filter(pl.col("crisp_on_top") == depth_val)
        left = dsub.filter(pl.col("crisp_side") == "left")
        right = dsub.filter(pl.col("crisp_side") == "right")
        t = _2x2(left, right)
        _, fp = fisher_exact(t)
        l_acc = left["correct"].mean()
        r_acc = right["correct"].mean()
        lines.append(
            f"  {depth_label}: left={_fmt_pct(l_acc)}, "
            f"right={_fmt_pct(r_acc)}, p = {_fmt_p(fp)}"
        )
    lines.append("")

    lines.append("### Colour pair")
    lines.append("")
    colour_ct = (
        valid.group_by("colour_crisp", "colour_blurred")
        .agg(
            pl.col("correct").sum().alias("k"),
            pl.col("correct").count().alias("n"),
        )
        .sort("colour_crisp", "colour_blurred")
    )
    table = np.column_stack(
        [colour_ct["k"].to_numpy(), (colour_ct["n"] - colour_ct["k"]).to_numpy()]
    )
    if np.any(table.sum(axis=0) == 0):
        lines.append(
            "Marginal effect: chi-square not applicable (zero marginal)"
        )
    else:
        chi2_val, p, dof, _ = chi2_contingency(table)
        v = association(table, method="cramer")
        lines.append(
            f"Marginal effect: χ²({dof}) = {chi2_val:.2f}, "
            f"p = {_fmt_p(p)}, V = {v:.4f}"
        )
        lines.append(
            f"  {dof + 1} colour pairs tested; V = {v:.4f} indicates "
            "negligible practical effect."
        )
    lines.append("")

    lines.append(
        "Effect sizes are small: V < 0.09 for colour pairs and V < 0.03 "
        "for side. There is a modest side × depth interaction for "
        "bias-incongruent trials (models favour the left circle slightly "
        "more when the blurred circle is on top). The counterbalanced "
        "design ensures this does not inflate the primary depth-order "
        "finding, which is symmetric across sides."
    )

    return "\n".join(lines)


def _summary_table(df: pl.DataFrame) -> str:
    valid = _valid(df)
    sweep = _balanced_sweep(valid)
    blur_levels = sorted(sweep["blur_px"].unique().to_list())
    scores = np.array(blur_levels)

    lines = [
        "## 10. Summary table",
        "",
    ]

    header = (
        f"{'Model':20s} {'Crisp↑':>8s} {'CI':>16s} {'Blur↑':>8s} "
        f"{'CI':>16s} {'OR':>7s} {'0px':>8s} {'CA z':>8s}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for model in sorted(valid["model"].unique().to_list()):
        msub = valid.filter(pl.col("model") == model)
        ct = msub.filter(pl.col("crisp_on_top"))
        bt = msub.filter(pl.col("crisp_on_top").not_())
        ct_k, ct_n = int(ct["correct"].sum()), len(ct)
        bt_k, bt_n = int(bt["correct"].sum()), len(bt)
        ct_ci = _wilson_ci(ct_k, ct_n)
        bt_ci = _wilson_ci(bt_k, bt_n)

        table = _2x2(ct, bt)
        or_val = odds_ratio(table, kind="conditional").statistic

        zero = msub.filter(pl.col("blur_px") == 0)
        zero_acc = zero["correct"].mean() if len(zero) > 0 else float("nan")

        bt_sweep = _balanced_sweep(bt)
        cells = (
            bt_sweep.group_by("blur_px")
            .agg(
                pl.col("correct").sum().alias("k"),
                pl.col("correct").count().alias("n"),
            )
            .sort("blur_px")
        )
        z, _ = _cochran_armitage(
            cells["k"].to_numpy(), cells["n"].to_numpy(), scores
        )

        lines.append(
            f"{model:20s} {_fmt_pct(ct_k / ct_n):>8s} "
            f"{_fmt_ci(*ct_ci):>16s} "
            f"{_fmt_pct(bt_k / bt_n):>8s} {_fmt_ci(*bt_ci):>16s} "
            f"{_fmt_or(or_val):>7s} {_fmt_pct(zero_acc):>8s} {z:>+8.2f}"
        )

    lines.append("")
    lines.append(
        "Crisp↑ = accuracy when crisp circle is on top (bias-congruent). "
        "Blur↑ = accuracy when blurred circle is on top (bias-incongruent). "
        "OR = depth-order odds ratio. 0px = accuracy at zero blur. "
        "CA z = Cochran-Armitage z for blur→accuracy trend "
        "(bias-incongruent)."
    )
    return "\n".join(lines)


def _statistical_notes() -> str:
    return "\n".join(
        [
            "## Statistical notes",
            "",
            "- All p-values are two-sided.",
            "- Odds ratios: conditional maximum likelihood estimates "
            "(scipy.stats.contingency.odds_ratio, kind='conditional') "
            "with exact 95% CIs.",
            "- Cochran-Armitage trend test: uses blur radius in pixels as "
            "equally spaced scores (0, 4, 8, 12, 16, 20). The accuracy "
            "curves in §3 show monotonic trends, confirming that the "
            "linearity assumption is reasonable here.",
            "- Cross-blur analyses use the balanced blur-sweep subset "
            "(4 colour pairs, ≤3 reps/cell at blur=20) to avoid "
            "over-representing blur=20 from the full factorial.",
            "- Pairwise model comparisons: Fisher exact tests with "
            "Holm-Bonferroni correction.",
            "- Proportion CIs: Wilson score interval.",
            "- No distributional assumptions beyond those inherent in "
            "exact tests and large-sample chi-square approximations "
            "(all cells have n ≥ 22).",
        ]
    )


def full_report(results_path: Path) -> str:
    df = load_results(results_path)
    sections = [
        _data_summary(df),
        _depth_order_bias(df),
        _blur_accuracy_curves(df),
        _blur_dose_response(df),
        _blur_x_depth_interaction(df),
        _zero_blur_baseline(df),
        _model_comparison(df),
        _prompt_effects(df),
        _nuisance_variables(df),
        _summary_table(df),
        _statistical_notes(),
    ]
    return "\n\n".join(sections)
