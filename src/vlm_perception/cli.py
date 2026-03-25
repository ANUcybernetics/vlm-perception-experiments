from pathlib import Path

import typer

from vlm_perception.evaluate import DEFAULT_PROMPT_ID, load_prompts
from vlm_perception.models import MODEL_REGISTRY

app = typer.Typer(help="VLM perception experiment: crisp vs blurred circle occlusion.")

DEFAULT_STIMULI_DIR = Path("stimuli")
DEFAULT_RESULTS_PATH = Path("results/results.csv")

AVAILABLE_MODELS = ", ".join(MODEL_REGISTRY)
AVAILABLE_PROMPTS = ", ".join(load_prompts())


@app.command()
def generate(
    output_dir: Path = typer.Option(
        DEFAULT_STIMULI_DIR, help="Directory for stimulus images"
    ),
) -> None:
    """Generate the full factorial set of stimulus images."""
    from vlm_perception.stimuli import generate_all

    paths = generate_all(output_dir)
    typer.echo(f"Generated {len(paths)} images in {output_dir}")


@app.command()
def evaluate(
    model: str = typer.Option(
        ..., help=f"Model name. Available: {AVAILABLE_MODELS}"
    ),
    reps: int = typer.Option(1, help="Number of repetitions per condition"),
    stimuli_dir: Path = typer.Option(
        DEFAULT_STIMULI_DIR, help="Directory containing stimulus images"
    ),
    results_path: Path = typer.Option(
        DEFAULT_RESULTS_PATH, help="CSV file for results"
    ),
    prompt: str = typer.Option(
        DEFAULT_PROMPT_ID, help=f"Prompt ID. Available: {AVAILABLE_PROMPTS}"
    ),
    limit: int = typer.Option(
        0, help="Max conditions to evaluate (0 = all)"
    ),
) -> None:
    """Run VLM evaluation on all stimulus images."""
    from vlm_perception.evaluate import evaluate as run_eval, get_prompt
    from vlm_perception.models import all_conditions, resolve_model
    from vlm_perception.storage import append_results

    get_prompt(prompt)
    spec = resolve_model(model)
    conditions = all_conditions()
    if limit > 0:
        conditions = conditions[:limit]
    total = len(conditions) * reps
    n = len(conditions)
    typer.echo(
        f"Running {total} trials ({n} conditions x {reps} reps) "
        f"with {model} ({spec.provider}/{spec.model_id}), prompt={prompt}"
    )

    results = []
    for rep in range(reps):
        for i, condition in enumerate(conditions):
            image_path = stimuli_dir / condition.image_filename
            if not image_path.exists():
                typer.echo(f"  Missing: {image_path}, skipping", err=True)
                continue
            trial_num = rep * len(conditions) + i + 1
            typer.echo(f"  [{trial_num}/{total}] {condition.image_filename}")
            result = run_eval(
                image_path, condition,
                provider=spec.provider, model=spec.model_id,
                prompt_id=prompt,
            )
            results.append(result)
            status = (
                "correct"
                if result.correct
                else ("incorrect" if result.correct is False else "unparseable")
            )
            typer.echo(f"    -> {result.parsed_answer} ({status})")

    append_results(results, results_path)
    n_correct = sum(1 for r in results if r.correct is True)
    n_total = sum(1 for r in results if r.correct is not None)
    typer.echo(
        f"\nDone. {n_correct}/{n_total} correct. Results saved to {results_path}"
    )


@app.command()
def analyse(
    results_path: Path = typer.Option(
        DEFAULT_RESULTS_PATH, help="CSV file with results"
    ),
) -> None:
    """Analyse results and print accuracy breakdowns."""
    from vlm_perception.analysis import full_report

    typer.echo(full_report(results_path))


if __name__ == "__main__":
    app()
