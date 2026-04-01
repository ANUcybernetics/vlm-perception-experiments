from pathlib import Path

import altair as alt
import polars as pl

from vlm_perception.analysis import _balanced_sweep, _valid
from vlm_perception.storage import load_results

FONT = "Libertinus Sans"


def _configure(chart: alt.Chart | alt.FacetChart) -> alt.Chart | alt.FacetChart:
    return chart.configure(
        font=FONT,
    ).configure_legend(
        orient="bottom",
        direction="horizontal",
    )


MODEL_ORDER = [
    "claude-haiku-4-5",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
]

MODEL_LABELS = {
    "claude-haiku-4-5": "Haiku 4.5",
    "claude-opus-4-6": "Opus 4.6",
    "claude-sonnet-4-6": "Sonnet 4.6",
    "gpt-5.4": "GPT-5.4",
    "gpt-5.4-mini": "GPT-5.4 Mini",
    "gpt-5.4-nano": "GPT-5.4 Nano",
}

DEPTH_LABELS = {True: "Crisp on top", False: "Blurred on top"}

PROMPT_ORDER = ["neutral", "minimal", "foreground", "psychophysics", "cot", "thinking"]

PROMPT_LABELS = {
    "neutral": "Neutral",
    "minimal": "Minimal",
    "foreground": "Foreground",
    "psychophysics": "Psychophysics",
    "cot": "CoT",
    "thinking": "Thinking",
}


def _prepare_dose_response(df: pl.DataFrame) -> pl.DataFrame:
    sweep = _balanced_sweep(_valid(df))
    grouped = (
        sweep.group_by("model", "blur_px", "crisp_on_top")
        .agg(
            pl.col("correct").sum().alias("k"),
            pl.col("correct").count().alias("n"),
        )
        .with_columns(
            (pl.col("k") / pl.col("n") * 100).alias("accuracy"),
            pl.col("model").replace(MODEL_LABELS).alias("model_label"),
            pl.when(pl.col("crisp_on_top"))
            .then(pl.lit("Crisp on top"))
            .otherwise(pl.lit("Blurred on top"))
            .alias("depth_order"),
        )
    )
    return grouped


def dose_response_chart(df: pl.DataFrame) -> alt.Chart:
    data = _prepare_dose_response(df)
    label_order = [MODEL_LABELS[m] for m in MODEL_ORDER if m in MODEL_LABELS]

    base = alt.Chart(data)

    chance = base.mark_rule(
        strokeDash=[4, 4], stroke="grey", strokeWidth=1
    ).encode(y=alt.datum(50))

    lines = base.mark_line(point=True).encode(
        x=alt.X("blur_px:Q", title="Blur radius (px)", scale=alt.Scale(domain=[0, 20])),
        y=alt.Y("accuracy:Q", title="Accuracy (%)", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color(
            "model_label:N",
            title="Model",
            sort=label_order,
        ),
    )

    chart = (
        (chance + lines)
        .properties(width=280, height=250)
        .facet(
            column=alt.Column(
                "depth_order:N",
                title=None,
                sort=["Crisp on top", "Blurred on top"],
                header=alt.Header(labelFontSize=13, labelFontWeight="bold"),
            ),
        )
    )
    return _configure(chart)


def _prepare_prompt_invariance(df: pl.DataFrame) -> pl.DataFrame:
    valid = _valid(df).filter(~pl.col("crisp_on_top"))
    grouped = (
        valid.group_by("prompt_id", "model")
        .agg(
            pl.col("correct").sum().alias("k"),
            pl.col("correct").count().alias("n"),
        )
        .with_columns(
            (pl.col("k") / pl.col("n") * 100).alias("accuracy"),
            pl.col("model").replace(MODEL_LABELS).alias("model_label"),
            pl.col("prompt_id").replace(PROMPT_LABELS).alias("prompt_label"),
        )
    )
    return grouped


def prompt_invariance_chart(df: pl.DataFrame) -> alt.Chart:
    data = _prepare_prompt_invariance(df)
    label_order = [MODEL_LABELS[m] for m in MODEL_ORDER if m in MODEL_LABELS]
    prompt_label_order = [PROMPT_LABELS[p] for p in PROMPT_ORDER]

    chance = alt.Chart(data).mark_rule(
        strokeDash=[4, 4], stroke="grey", strokeWidth=1
    ).encode(y=alt.datum(50))

    lines = alt.Chart(data).mark_line(point=True).encode(
        x=alt.X(
            "prompt_label:N",
            title="Prompt variant",
            sort=prompt_label_order,
            axis=alt.Axis(labelAngle=-30),
        ),
        y=alt.Y(
            "accuracy:Q",
            title="Accuracy (%, bias-incongruent)",
            scale=alt.Scale(domain=[0, 60]),
        ),
        color=alt.Color(
            "model_label:N",
            title="Model",
            sort=label_order,
            scale=alt.Scale(scheme="dark2"),
        ),
    )

    return _configure((chance + lines).properties(width=350, height=250))


def save_chart(chart: alt.Chart, output: Path) -> None:
    suffix = output.suffix.lower()
    if suffix == ".pdf":
        chart.save(str(output), format="pdf")
    elif suffix == ".png":
        chart.save(str(output), format="png", scale_factor=2)
    elif suffix == ".svg":
        chart.save(str(output), format="svg")
    else:
        msg = f"Unsupported format: {suffix}"
        raise ValueError(msg)


def generate_figures(results_path: Path, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_results(results_path)

    paths = []

    dose_path = output_dir / "dose_response.pdf"
    chart = dose_response_chart(df)
    save_chart(chart, dose_path)
    paths.append(dose_path)

    prompt_path = output_dir / "prompt_invariance.pdf"
    chart = prompt_invariance_chart(df)
    save_chart(chart, prompt_path)
    paths.append(prompt_path)

    return paths
