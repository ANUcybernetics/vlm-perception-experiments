import asyncio
from pathlib import Path

import typer

from vlm_perception.evaluate import DEFAULT_PROMPT_ID, load_prompts
from vlm_perception.models import MODEL_REGISTRY

app = typer.Typer(
    help="VLM perception experiment: crisp vs blurred circle occlusion."
)

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
    limit: int = typer.Option(
        0, help="Max conditions to evaluate (0 = all)"
    ),
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
                    trials.append(
                        (m, pid, specs[m].provider, 0, ci)
                    )

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
                else (
                    "incorrect"
                    if result.correct is False
                    else "unparseable"
                )
            )
            typer.echo(
                f"  [{completed}/{total}] {model_name} {prompt_id} "
                f"{condition.image_filename} "
                f"-> {result.parsed_answer} ({status})"
            )

    tasks = [
        run_trial(m, pid, prov, ci)
        for m, pid, prov, _rep, ci in trials
    ]
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


if __name__ == "__main__":
    app()
