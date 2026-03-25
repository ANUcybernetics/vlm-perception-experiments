from pathlib import Path

import polars as pl

from vlm_perception.storage import load_results


def overall_accuracy(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.filter(pl.col("correct").is_not_null())
        .group_by("model")
        .agg(
            pl.col("correct").sum().alias("n_correct"),
            pl.col("correct").count().alias("n_total"),
            pl.col("correct").mean().alias("accuracy"),
        )
        .sort("model")
    )


def accuracy_by_layout(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.filter(pl.col("correct").is_not_null())
        .group_by("model", "crisp_on_top")
        .agg(
            pl.col("correct").sum().alias("n_correct"),
            pl.col("correct").count().alias("n_total"),
            pl.col("correct").mean().alias("accuracy"),
        )
        .sort("model", "crisp_on_top")
    )


def accuracy_by_colour(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.filter(pl.col("correct").is_not_null())
        .group_by("model", "colour_crisp", "colour_blurred")
        .agg(
            pl.col("correct").sum().alias("n_correct"),
            pl.col("correct").count().alias("n_total"),
            pl.col("correct").mean().alias("accuracy"),
        )
        .sort("model", "colour_crisp", "colour_blurred")
    )


def accuracy_by_side(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.filter(pl.col("correct").is_not_null())
        .group_by("model", "crisp_side")
        .agg(
            pl.col("correct").sum().alias("n_correct"),
            pl.col("correct").count().alias("n_total"),
            pl.col("correct").mean().alias("accuracy"),
        )
        .sort("model", "crisp_side")
    )


def unparseable_count(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.filter(pl.col("parsed_answer").is_null())
        .group_by("model")
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

    sections.append("\n## Accuracy by colour pair")
    sections.append(str(accuracy_by_colour(df)))

    unp = unparseable_count(df)
    if len(unp) > 0:
        sections.append("\n## Unparseable responses")
        sections.append(str(unp))

    return "\n".join(sections)
