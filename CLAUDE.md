# vlm-perception

VLM perception experiment: testing whether vision-language models can correctly identify occlusion order of crisp vs blurred overlapping circles.

## Project structure

- `src/vlm_perception/` --- main package
  - `models.py` --- pydantic models (Colour, Side, Condition, TrialResult) and factorial condition generation
  - `stimuli.py` --- Pillow-based stimulus image generation
  - `evaluate.py` --- VLM API dispatch (Anthropic, OpenAI) with response parsing
  - `storage.py` --- CSV I/O via polars
  - `analysis.py` --- accuracy breakdowns by condition
  - `cli.py` --- typer CLI entry point
- `tests/` --- pytest tests

## Commands

- `mise exec -- uv sync` --- install dependencies
- `mise exec -- uv run vlm-perception generate` --- generate stimulus images
- `mise exec -- uv run vlm-perception evaluate --provider anthropic --model claude-opus-4-6-20250415` --- run evaluation
- `mise exec -- uv run vlm-perception analyse` --- print analysis
- `mise exec -- uv run pytest -x` --- run tests

## Conventions

- Use `uv` via mise for all Python tooling
- polars for dataframes, pydantic for validation
- Australian English in prose (colour, analyse, etc.)
- Results stored as CSV in `results/`
- Stimulus images in `stimuli/` (both gitignored)
