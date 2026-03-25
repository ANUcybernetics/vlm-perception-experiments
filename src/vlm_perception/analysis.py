from pathlib import Path

import polars as pl
from scipy.stats import chi2_contingency, fisher_exact

from vlm_perception.storage import load_results

GROUP = ["model", "prompt_id"]


def overall_accuracy(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.filter(pl.col("correct").is_not_null())
        .group_by(*GROUP)
        .agg(
            pl.col("correct").sum().alias("n_correct"),
            pl.col("correct").count().alias("n_total"),
            pl.col("correct").mean().alias("accuracy"),
        )
        .sort(*GROUP)
    )


def accuracy_by_layout(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.filter(pl.col("correct").is_not_null())
        .group_by(*GROUP, "crisp_on_top")
        .agg(
            pl.col("correct").sum().alias("n_correct"),
            pl.col("correct").count().alias("n_total"),
            pl.col("correct").mean().alias("accuracy"),
        )
        .sort(*GROUP, "crisp_on_top")
    )


def accuracy_by_colour(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.filter(pl.col("correct").is_not_null())
        .group_by(*GROUP, "colour_crisp", "colour_blurred")
        .agg(
            pl.col("correct").sum().alias("n_correct"),
            pl.col("correct").count().alias("n_total"),
            pl.col("correct").mean().alias("accuracy"),
        )
        .sort(*GROUP, "colour_crisp", "colour_blurred")
    )


def accuracy_by_side(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.filter(pl.col("correct").is_not_null())
        .group_by(*GROUP, "crisp_side")
        .agg(
            pl.col("correct").sum().alias("n_correct"),
            pl.col("correct").count().alias("n_total"),
            pl.col("correct").mean().alias("accuracy"),
        )
        .sort(*GROUP, "crisp_side")
    )


def depth_order_fisher_test(df: pl.DataFrame) -> pl.DataFrame:
    """Fisher's exact test for association between depth order and correctness.

    For each model/prompt combination, constructs a 2x2 contingency table
    (crisp-on-top vs blurred-on-top) x (correct vs incorrect) and runs
    Fisher's exact test. This is the right test here: it's exact (no
    asymptotic approximation), makes no distributional assumptions, and
    directly tests whether depth order is associated with response correctness.
    """
    valid = df.filter(pl.col("correct").is_not_null())
    rows = []
    groups = valid.select(*GROUP).unique().sort(*GROUP)
    for row in groups.iter_rows(named=True):
        mask = pl.lit(True)
        for col in GROUP:
            mask = mask & (pl.col(col) == row[col])
        m = valid.filter(mask)
        crisp_top = m.filter(pl.col("crisp_on_top"))
        blurred_top = m.filter(pl.col("crisp_on_top").not_())
        table = [
            [int(crisp_top["correct"].sum()), int((~crisp_top["correct"]).sum())],
            [int(blurred_top["correct"].sum()), int((~blurred_top["correct"]).sum())],
        ]
        odds_ratio, p_value = fisher_exact(table)
        rows.append({**row, "odds_ratio": odds_ratio, "p_value": p_value})
    return pl.DataFrame(rows).sort(*GROUP)


def prompt_effect(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Chi-square test for prompt effect on correctness.

    Returns (per_model results, pooled result) where each row contains
    chi2, dof, and p_value. Also includes a wide accuracy table.
    """
    valid = df.filter(pl.col("correct").is_not_null())

    def _chi2_for(sub: pl.DataFrame) -> dict:
        ct = (
            sub.group_by("prompt_id")
            .agg(
                pl.col("correct").sum().alias("correct"),
                (pl.col("correct").count() - pl.col("correct").sum()).alias("incorrect"),
            )
            .sort("prompt_id")
        )
        table = ct.select("correct", "incorrect").to_numpy()
        chi2, p, dof, _ = chi2_contingency(table)
        return {"chi2": chi2, "dof": dof, "p_value": p}

    per_model_rows = []
    for model in sorted(valid["model"].unique().to_list()):
        sub = valid.filter(pl.col("model") == model)
        per_model_rows.append({"model": model, **_chi2_for(sub)})
    per_model = pl.DataFrame(per_model_rows)

    pooled = pl.DataFrame([{"model": "(pooled)", **_chi2_for(valid)}])

    wide = (
        valid.group_by("model", "prompt_id")
        .agg(pl.col("correct").mean().alias("accuracy"))
        .pivot(on="prompt_id", index="model", values="accuracy")
        .sort("model")
    )

    return pl.concat([per_model, pooled]), wide


def unparseable_count(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.filter(pl.col("parsed_answer").is_null())
        .group_by(*GROUP)
        .agg(pl.col("model").count().alias("n_unparseable"))
    )


def full_report(results_path: Path) -> str:
    df = load_results(results_path)
    sections = []

    sections.append("## Overall accuracy")
    sections.append(str(overall_accuracy(df)))

    sections.append("\n## Accuracy by layout (crisp on top vs blurred on top)")
    sections.append(str(accuracy_by_layout(df)))

    sections.append("\n## Accuracy by crisp circle side")
    sections.append(str(accuracy_by_side(df)))

    sections.append("\n## Fisher's exact test: depth order vs correctness")
    sections.append(str(depth_order_fisher_test(df)))

    chi2_results, wide_acc = prompt_effect(df)
    sections.append("\n## Prompt effect: accuracy by model x prompt")
    sections.append(str(wide_acc))
    sections.append("\n## Prompt effect: chi-square test (prompt x correctness)")
    sections.append(str(chi2_results))

    sections.append("\n## Accuracy by colour pair")
    sections.append(str(accuracy_by_colour(df)))

    unp = unparseable_count(df)
    if len(unp) > 0:
        sections.append("\n## Unparseable responses")
        sections.append(str(unp))

    return "\n".join(sections)
