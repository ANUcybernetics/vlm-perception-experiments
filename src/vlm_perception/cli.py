from pathlib import Path

import typer

app = typer.Typer(help="VLM perception experiment: crisp vs blurred circle occlusion.")

DEFAULT_STIMULI_DIR = Path("stimuli")
DEFAULT_RESULTS_PATH = Path("results/results.csv")


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
    provider: str = typer.Option(..., help="API provider: anthropic or openai"),
    model: str = typer.Option(
        ..., help="Model identifier (e.g. claude-opus-4-6-20250415, gpt-4o)"
    ),
    reps: int = typer.Option(1, help="Number of repetitions per condition"),
    stimuli_dir: Path = typer.Option(
        DEFAULT_STIMULI_DIR, help="Directory containing stimulus images"
    ),
    results_path: Path = typer.Option(
        DEFAULT_RESULTS_PATH, help="CSV file for results"
    ),
) -> None:
    """Run VLM evaluation on all stimulus images."""
    from vlm_perception.evaluate import evaluate as run_eval
    from vlm_perception.models import all_conditions
    from vlm_perception.storage import append_results

    conditions = all_conditions()
    total = len(conditions) * reps
    n = len(conditions)
    typer.echo(f"Running {total} trials ({n} conditions x {reps} reps) with {model}")

    results = []
    for rep in range(reps):
        for i, condition in enumerate(conditions):
            image_path = stimuli_dir / condition.image_filename
            if not image_path.exists():
                typer.echo(f"  Missing: {image_path}, skipping", err=True)
                continue
            trial_num = rep * len(conditions) + i + 1
            typer.echo(f"  [{trial_num}/{total}] {condition.image_filename}")
            result = run_eval(image_path, condition, provider=provider, model=model)
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
