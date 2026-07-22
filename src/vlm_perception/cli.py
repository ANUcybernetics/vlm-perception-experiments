import asyncio
from pathlib import Path

import typer

from vlm_perception.evaluate import DEFAULT_PROMPT_ID, load_prompts
from vlm_perception.models import MODEL_REGISTRY

app = typer.Typer(help="VLM perception experiment: crisp vs blurred circle occlusion.")

DEFAULT_STIMULI_DIR = Path("stimuli")
DEFAULT_RESULTS_PATH = Path("results/results.jsonl")
DEFAULT_CONCURRENCY = 10

AVAILABLE_MODELS = ", ".join(MODEL_REGISTRY)
AVAILABLE_PROMPTS = ", ".join(load_prompts())


@app.command()
def generate(
    output_dir: Path = typer.Option(
        DEFAULT_STIMULI_DIR, help="Directory for stimulus images"
    ),
    blur_sweep: bool = typer.Option(
        False,
        help="Generate reduced stimulus set for blur radius sweep",
    ),
) -> None:
    """Generate stimulus images."""
    from vlm_perception.models import all_conditions, blur_sweep_conditions
    from vlm_perception.stimuli import generate_all

    conditions = blur_sweep_conditions() if blur_sweep else all_conditions()
    paths = generate_all(output_dir, conditions)
    typer.echo(f"Generated {len(paths)} images in {output_dir}")


@app.command()
def evaluate(
    model: list[str] = typer.Option(
        ..., help=f"Model name(s). Available: {AVAILABLE_MODELS}"
    ),
    reps: int = typer.Option(1, help="Number of repetitions per condition"),
    stimuli_dir: Path = typer.Option(
        DEFAULT_STIMULI_DIR, help="Directory containing stimulus images"
    ),
    results_path: Path = typer.Option(
        DEFAULT_RESULTS_PATH, help="JSONL file for results"
    ),
    prompt: list[str] = typer.Option(
        [DEFAULT_PROMPT_ID],
        help=f"Prompt ID(s). Available: {AVAILABLE_PROMPTS}",
    ),
    limit: int = typer.Option(0, help="Max conditions to evaluate (0 = all)"),
    concurrency: int = typer.Option(
        DEFAULT_CONCURRENCY, help="Max concurrent requests per provider"
    ),
    blur_sweep: bool = typer.Option(
        False,
        help="Use reduced blur sweep conditions (80) instead of full factorial (120)",
    ),
    resume: bool = typer.Option(
        False,
        help="Skip trials already present in results file",
    ),
) -> None:
    """Run VLM evaluation on stimulus images."""
    asyncio.run(
        _evaluate_async(
            models=model,
            prompt_ids=prompt,
            reps=reps,
            stimuli_dir=stimuli_dir,
            results_path=results_path,
            limit=limit,
            concurrency=concurrency,
            blur_sweep=blur_sweep,
            resume=resume,
        )
    )


async def _evaluate_async(
    models: list[str],
    prompt_ids: list[str],
    reps: int,
    stimuli_dir: Path,
    results_path: Path,
    limit: int,
    concurrency: int,
    blur_sweep: bool,
    resume: bool,
) -> None:
    from vlm_perception.evaluate import async_evaluate, get_prompt
    from vlm_perception.models import (
        all_conditions,
        blur_sweep_conditions,
        resolve_model,
    )
    from vlm_perception.storage import (
        async_append_result,
        existing_trial_counts,
    )

    for pid in prompt_ids:
        get_prompt(pid)
    specs = {m: resolve_model(m) for m in models}

    conditions = blur_sweep_conditions() if blur_sweep else all_conditions()
    if limit > 0:
        conditions = conditions[:limit]

    semaphores: dict[str, asyncio.Semaphore] = {}
    for m in models:
        provider = specs[m].provider
        if provider not in semaphores:
            semaphores[provider] = asyncio.Semaphore(concurrency)

    existing = existing_trial_counts(results_path) if resume else {}

    trials: list[tuple[str, str, str, int, int]] = []
    skipped = 0
    for m in models:
        for pid in prompt_ids:
            for ci, condition in enumerate(conditions):
                key = (
                    specs[m].model_id,
                    pid,
                    condition.blur_radius,
                    condition.crisp_on_top,
                    condition.crisp_side.value,
                    condition.colour_crisp.value,
                    condition.colour_blurred.value,
                )
                already_done = existing.get(key, 0)
                needed = max(0, reps - already_done)
                skipped += reps - needed
                for _rep in range(needed):
                    trials.append((m, pid, specs[m].provider, 0, ci))

    total = len(trials)
    if resume and skipped > 0:
        typer.echo(f"Resuming: skipping {skipped} already-completed trials")
    if total == 0:
        typer.echo("All trials already completed, nothing to do.")
        return
    typer.echo(
        f"Running {total} trials "
        f"({len(models)} model(s) x {len(prompt_ids)} prompt(s) x "
        f"{len(conditions)} conditions x {reps} reps, "
        f"concurrency={concurrency}/provider)"
    )

    file_lock = asyncio.Lock()
    counter_lock = asyncio.Lock()
    completed = 0
    n_correct = 0
    n_total = 0
    n_errors = 0

    async def run_trial(
        model_name: str,
        prompt_id: str,
        provider: str,
        condition_idx: int,
    ) -> None:
        nonlocal completed, n_correct, n_total, n_errors
        condition = conditions[condition_idx]
        image_path = stimuli_dir / condition.image_filename
        if not image_path.exists():
            async with counter_lock:
                completed += 1
            typer.echo(
                f"  [{completed}/{total}] MISSING: {image_path}",
                err=True,
            )
            return

        try:
            result = await async_evaluate(
                image_path,
                condition,
                provider=provider,
                model=specs[model_name].model_id,
                prompt_id=prompt_id,
                semaphore=semaphores[provider],
            )
        except Exception as exc:
            async with counter_lock:
                completed += 1
                n_errors += 1
            typer.echo(
                f"  [{completed}/{total}] ERROR: {model_name} "
                f"{prompt_id} {condition.image_filename} "
                f"-> {type(exc).__name__}: {exc}",
                err=True,
            )
            return

        await async_append_result(result, results_path, file_lock)

        async with counter_lock:
            completed += 1
            if result.correct is not None:
                n_total += 1
                if result.correct:
                    n_correct += 1
            status = (
                "correct"
                if result.correct
                else ("incorrect" if result.correct is False else "unparseable")
            )
            typer.echo(
                f"  [{completed}/{total}] {model_name} {prompt_id} "
                f"{condition.image_filename} "
                f"-> {result.parsed_answer} ({status})"
            )

    tasks = [run_trial(m, pid, prov, ci) for m, pid, prov, _rep, ci in trials]
    await asyncio.gather(*tasks)

    error_msg = f" ({n_errors} errors)" if n_errors else ""
    typer.echo(
        f"\nDone. {n_correct}/{n_total} correct{error_msg}. "
        f"Results saved to {results_path}"
    )


@app.command()
def analyse(
    results_path: Path = typer.Option(
        DEFAULT_RESULTS_PATH, help="JSONL file with results"
    ),
) -> None:
    """Analyse results and print accuracy breakdowns."""
    from vlm_perception.analysis import full_report

    typer.echo(full_report(results_path))


DEFAULT_FIGURES_DIR = Path("figures")
DEFAULT_JUDGMENTS_PATH = Path("results/judgments.jsonl")


@app.command()
def judge(
    results_path: Path = typer.Option(
        DEFAULT_RESULTS_PATH, help="JSONL file with results to judge"
    ),
    output_path: Path = typer.Option(
        DEFAULT_JUDGMENTS_PATH, help="JSONL file for trace judgments"
    ),
    limit: int = typer.Option(0, help="Max traces to judge (0 = all)"),
    concurrency: int = typer.Option(8, help="Max concurrent Anthropic requests"),
    include_bias_congruent: bool = typer.Option(
        False,
        help="Also judge bias-congruent traces (default: incongruent only)",
    ),
) -> None:
    """Run LLM-as-judge categorisation of MLLM reasoning traces.

    Uses Claude Sonnet 4.6 to label each trace along seven boolean
    dimensions related to the verbal heuristic (sharp = closer) hypothesis.
    See `src/vlm_perception/judge.py` for label definitions and the
    self-judgment-bias caveat.
    """
    from vlm_perception.judge import run_judge

    run_judge(
        results_path,
        output_path,
        limit=limit if limit > 0 else None,
        concurrency=concurrency,
        only_bias_incongruent=not include_bias_congruent,
    )
    typer.echo(f"Judgments written to {output_path}")


@app.command()
def analyse_judgments(
    judgments_path: Path = typer.Option(
        DEFAULT_JUDGMENTS_PATH, help="JSONL file with trace judgments"
    ),
) -> None:
    """Print per-model trace-judgment label frequencies."""
    from vlm_perception.judge import judgment_summary

    typer.echo(judgment_summary(judgments_path))


@app.command()
def plot(
    results_path: Path = typer.Option(
        DEFAULT_RESULTS_PATH, help="JSONL file with results"
    ),
    output_dir: Path = typer.Option(
        DEFAULT_FIGURES_DIR, help="Directory for output figures"
    ),
) -> None:
    """Generate figures from results."""
    from vlm_perception.plotting import generate_figures

    paths = generate_figures(results_path, output_dir)
    for p in paths:
        typer.echo(f"Saved {p}")


scotoma_app = typer.Typer(
    help="Scotoma: dual-stream typeface exploiting occlusion edge blur."
)
app.add_typer(scotoma_app, name="scotoma")


@scotoma_app.command("render")
def scotoma_render(
    real: str = typer.Option(
        None, help="Real (human-readable) text; literal \\n breaks lines"
    ),
    robot: str = typer.Option(None, help="Robot text (same printable length)"),
    real_file: Path = typer.Option(None, help="Read real text from file"),
    robot_file: Path = typer.Option(None, help="Read robot text from file"),
    output: Path = typer.Option(Path("scotoma.png"), "--output", "-o"),
    font_size: int = typer.Option(96, help="Font size in px"),
    blur_fraction: float = typer.Option(
        0.07, help="Gaussian blur radius as fraction of font size (0 = crisp)"
    ),
    offset_fraction: float = typer.Option(
        0.38, help="Total horizontal layer separation as fraction of font size"
    ),
    colour_real: str = typer.Option("red", help="Colour of the real stream"),
    colour_robot: str = typer.Option("cyan", help="Colour of the robot stream"),
    background_grey: int = typer.Option(128, help="Background grey level 0-255"),
    crisp_on_top: bool = typer.Option(
        False, help="Composite the crisp robot layer in front (congruent control)"
    ),
) -> None:
    """Render two text streams as a Scotoma image."""
    from vlm_perception.models import Colour
    from vlm_perception.scotoma import ScotomaStyle, render_scotoma

    if (real is None) == (real_file is None):
        raise typer.BadParameter("Provide exactly one of --real / --real-file")
    if (robot is None) == (robot_file is None):
        raise typer.BadParameter("Provide exactly one of --robot / --robot-file")

    real_text = real_file.read_text() if real_file else real.replace("\\n", "\n")
    robot_text = robot_file.read_text() if robot_file else robot

    style = ScotomaStyle(
        font_size=font_size,
        blur_fraction=blur_fraction,
        offset_fraction=offset_fraction,
        colour_real=Colour(colour_real),
        colour_robot=Colour(colour_robot),
        background_grey=background_grey,
        blurred_on_top=not crisp_on_top,
    )
    img = render_scotoma(real_text, robot_text, style)
    output.parent.mkdir(parents=True, exist_ok=True)
    img.save(output)
    typer.echo(f"Saved {output} ({img.width}x{img.height})")


@scotoma_app.command("diptych")
def scotoma_diptych(
    top: str = typer.Option(
        ..., help="First message (human reads it in the top panel)"
    ),
    bottom: str = typer.Option(
        ..., help="Second message (human reads it in the bottom panel)"
    ),
    output: Path = typer.Option(Path("scotoma-diptych.png"), "--output", "-o"),
    font_size: int = typer.Option(96, help="Font size in px"),
    blur_fraction: float = typer.Option(
        0.07, help="Blur radius as fraction of font size"
    ),
    offset_fraction: float = typer.Option(
        0.38, help="Horizontal layer separation as fraction of font size"
    ),
    colour_real: str = typer.Option("red", help="Colour of the human (blurred) stream"),
    colour_robot: str = typer.Option("cyan", help="Colour of the VLM (crisp) stream"),
    background_grey: int = typer.Option(128, help="Background grey level 0-255"),
) -> None:
    """Render the reciprocal diptych: two panels that swap who reads what."""
    from vlm_perception.models import Colour
    from vlm_perception.scotoma import ScotomaStyle, render_diptych

    style = ScotomaStyle(
        font_size=font_size,
        blur_fraction=blur_fraction,
        offset_fraction=offset_fraction,
        colour_real=Colour(colour_real),
        colour_robot=Colour(colour_robot),
        background_grey=background_grey,
    )
    img = render_diptych(top, bottom, style)
    output.parent.mkdir(parents=True, exist_ok=True)
    img.save(output)
    typer.echo(f"Saved {output} ({img.width}x{img.height})")


DEFAULT_SCOTOMA_STIMULI_DIR = Path("stimuli/scotoma")
DEFAULT_SCOTOMA_RESULTS_PATH = Path("results/scotoma.jsonl")


@scotoma_app.command("generate")
def scotoma_generate(
    output_dir: Path = typer.Option(
        DEFAULT_SCOTOMA_STIMULI_DIR, help="Directory for stimulus images"
    ),
    font_size: int = typer.Option(96, help="Font size in px"),
) -> None:
    """Generate the Scotoma experiment stimuli (main sweep + legibility solo)."""
    from vlm_perception.scotoma.experiment import (
        generate_stimuli,
        legibility_conditions,
        scotoma_conditions,
    )

    conditions = scotoma_conditions(font_size) + legibility_conditions(font_size)
    paths = generate_stimuli(output_dir, conditions)
    typer.echo(f"Generated {len(paths)} images in {output_dir}")


@scotoma_app.command("evaluate")
def scotoma_evaluate(
    model: list[str] = typer.Option(
        ..., help=f"Model name(s). Available: {AVAILABLE_MODELS}"
    ),
    prompt: list[str] = typer.Option(
        ["naive"], help="Prompt ID(s). Available: naive, dual, cot, thinking"
    ),
    reps: int = typer.Option(1, help="Number of repetitions per condition"),
    stimuli_dir: Path = typer.Option(
        DEFAULT_SCOTOMA_STIMULI_DIR, help="Directory containing stimulus images"
    ),
    results_path: Path = typer.Option(
        DEFAULT_SCOTOMA_RESULTS_PATH, help="JSONL file for results"
    ),
    font_size: int = typer.Option(96, help="Font size of the stimuli to evaluate"),
    legibility: bool = typer.Option(
        False, help="Evaluate the solo-string legibility baseline instead"
    ),
    limit: int = typer.Option(0, help="Max conditions to evaluate (0 = all)"),
    concurrency: int = typer.Option(
        DEFAULT_CONCURRENCY, help="Max concurrent requests per provider"
    ),
    resume: bool = typer.Option(
        False, help="Skip trials already present in results file"
    ),
) -> None:
    """Run VLM transcription evaluation on Scotoma stimuli."""
    asyncio.run(
        _scotoma_evaluate_async(
            models=model,
            prompt_ids=prompt,
            reps=reps,
            stimuli_dir=stimuli_dir,
            results_path=results_path,
            font_size=font_size,
            legibility=legibility,
            limit=limit,
            concurrency=concurrency,
            resume=resume,
        )
    )


async def _scotoma_evaluate_async(
    models: list[str],
    prompt_ids: list[str],
    reps: int,
    stimuli_dir: Path,
    results_path: Path,
    font_size: int,
    legibility: bool,
    limit: int,
    concurrency: int,
    resume: bool,
) -> None:
    from vlm_perception.models import resolve_model
    from vlm_perception.scotoma.evaluate import async_evaluate_scotoma, get_prompt
    from vlm_perception.scotoma.experiment import (
        legibility_conditions,
        scotoma_conditions,
    )
    from vlm_perception.scotoma.storage import (
        async_append_result,
        existing_trial_counts,
    )

    for pid in prompt_ids:
        get_prompt(pid)
    specs = {m: resolve_model(m) for m in models}

    conditions = (
        legibility_conditions(font_size)
        if legibility
        else scotoma_conditions(font_size)
    )
    if limit > 0:
        conditions = conditions[:limit]

    semaphores: dict[str, asyncio.Semaphore] = {}
    for m in models:
        provider = specs[m].provider
        if provider not in semaphores:
            semaphores[provider] = asyncio.Semaphore(concurrency)

    existing = existing_trial_counts(results_path) if resume else {}

    trials: list[tuple[str, str, str, int]] = []
    skipped = 0
    for m in models:
        for pid in prompt_ids:
            for ci, condition in enumerate(conditions):
                key = (
                    specs[m].model_id,
                    pid,
                    condition.string_real,
                    condition.string_robot,
                    condition.blur_fraction,
                    condition.blurred_on_top,
                    condition.colour_real.value,
                    condition.font_size,
                )
                already_done = existing.get(key, 0)
                needed = max(0, reps - already_done)
                skipped += reps - needed
                for _rep in range(needed):
                    trials.append((m, pid, specs[m].provider, ci))

    total = len(trials)
    if resume and skipped > 0:
        typer.echo(f"Resuming: skipping {skipped} already-completed trials")
    if total == 0:
        typer.echo("All trials already completed, nothing to do.")
        return
    typer.echo(
        f"Running {total} trials "
        f"({len(models)} model(s) x {len(prompt_ids)} prompt(s) x "
        f"{len(conditions)} conditions x {reps} reps, "
        f"concurrency={concurrency}/provider)"
    )

    file_lock = asyncio.Lock()
    counter_lock = asyncio.Lock()
    completed = 0
    n_errors = 0

    async def run_trial(
        model_name: str, prompt_id: str, provider: str, condition_idx: int
    ) -> None:
        nonlocal completed, n_errors
        condition = conditions[condition_idx]
        image_path = stimuli_dir / condition.image_filename
        if not image_path.exists():
            async with counter_lock:
                completed += 1
            typer.echo(f"  [{completed}/{total}] MISSING: {image_path}", err=True)
            return

        try:
            result = await async_evaluate_scotoma(
                image_path,
                condition,
                provider=provider,
                model=specs[model_name].model_id,
                prompt_id=prompt_id,
                semaphore=semaphores[provider],
            )
        except Exception as exc:
            async with counter_lock:
                completed += 1
                n_errors += 1
            typer.echo(
                f"  [{completed}/{total}] ERROR: {model_name} "
                f"{prompt_id} {condition.image_filename} "
                f"-> {type(exc).__name__}: {exc}",
                err=True,
            )
            return

        await async_append_result(result, results_path, file_lock)

        async with counter_lock:
            completed += 1
            bias = (
                f"bias={result.bias_index_lev:+.2f}"
                if result.bias_index_lev is not None
                else f"d_real={result.dist_real_lev}"
            )
            typer.echo(
                f"  [{completed}/{total}] {model_name} {prompt_id} "
                f"{condition.image_filename} "
                f"-> {result.raw_transcription} ({bias})"
            )

    tasks = [run_trial(m, pid, prov, ci) for m, pid, prov, ci in trials]
    await asyncio.gather(*tasks)

    error_msg = f" ({n_errors} errors)" if n_errors else ""
    typer.echo(f"\nDone{error_msg}. Results saved to {results_path}")


@scotoma_app.command("analyse")
def scotoma_analyse(
    results_path: Path = typer.Option(
        DEFAULT_SCOTOMA_RESULTS_PATH, help="JSONL file with Scotoma results"
    ),
) -> None:
    """Analyse Scotoma transcription results."""
    from vlm_perception.scotoma.analysis import full_report

    typer.echo(full_report(results_path))


@scotoma_app.command("precheck")
def scotoma_precheck(
    model: list[str] = typer.Option(
        ..., help=f"Model name(s). Available: {AVAILABLE_MODELS}"
    ),
    font_size: list[int] = typer.Option(
        [48, 72, 96], help="Candidate font sizes to test"
    ),
    stimuli_dir: Path = typer.Option(
        Path("stimuli/scotoma-precheck"), help="Directory for precheck images"
    ),
    results_path: Path = typer.Option(
        Path("results/scotoma-precheck.jsonl"), help="JSONL file for precheck results"
    ),
    concurrency: int = typer.Option(DEFAULT_CONCURRENCY),
) -> None:
    """Resolution pre-check: solo-string legibility at candidate font sizes.

    Verifies the chosen font/canvas size survives provider-side image
    downsampling before the main sweep is run.
    """
    import polars as pl

    from vlm_perception.scotoma.experiment import (
        generate_stimuli,
        legibility_conditions,
    )
    from vlm_perception.scotoma.storage import load_results

    for fs in font_size:
        conditions = legibility_conditions(fs)
        generate_stimuli(stimuli_dir, conditions)
        asyncio.run(
            _scotoma_evaluate_async(
                models=model,
                prompt_ids=["naive"],
                reps=1,
                stimuli_dir=stimuli_dir,
                results_path=results_path,
                font_size=fs,
                legibility=True,
                limit=0,
                concurrency=concurrency,
                resume=True,
            )
        )

    df = load_results(results_path)
    table = (
        df.group_by("model", "font_size")
        .agg(
            pl.len().alias("n"),
            (pl.col("dist_real_lev") == 0.0).mean().round(3).alias("exact_rate"),
            pl.col("dist_real_lev").mean().round(4).alias("mean_dist_lev"),
        )
        .sort("model", "font_size")
    )
    typer.echo(table)


if __name__ == "__main__":
    app()
