# vlm-perception

VLM perception experiment: testing whether vision-language models can correctly identify occlusion order of crisp vs blurred overlapping circles.

## Quick start

```sh
mise install          # install uv
uv sync               # install Python dependencies
uv run vlm-perception generate  # generate 120 stimulus images in stimuli/
```

To run VLM evaluation, set the relevant API key and run:

```sh
export ANTHROPIC_API_KEY="..."
uv run vlm-perception evaluate --provider anthropic --model claude-opus-4-6-20250415 --reps 3

export OPENAI_API_KEY="..."
uv run vlm-perception evaluate --provider openai --model gpt-4o --reps 3
```

Results append to `results/results.csv`. Analyse with:

```sh
uv run vlm-perception analyse
```

Run tests with:

```sh
uv run pytest -x
```

## Experimental design

Each stimulus image shows two overlapping circles on a mid-grey background. One circle is crisp, the other is Gaussian-blurred. The question posed to the VLM is: which circle is in front (occluding the other)?

### Factorial conditions (120 total)

- **depth order** (2): crisp on top, blurred on top
- **spatial position** (2): crisp circle on left, crisp circle on right
- **colour pairs** (6 x 5 = 30): 6 hues (OKLCH, perceptually uniform) for each circle, excluding same-colour pairs

### Stimulus parameters

- canvas: 512x512px, background RGB (128, 128, 128)
- circle radius: 100px, centre offset: 75px (~25% area overlap)
- blur: Gaussian radius 8px
- colours: 6 OKLCH hues at L=0.7, C=0.15 (red, yellow, green, cyan, blue, magenta)

## Project structure

- `src/vlm_perception/` --- main package
  - `models.py` --- pydantic models (Colour, Side, Condition, TrialResult) and `all_conditions()` factorial generator
  - `stimuli.py` --- Pillow-based stimulus image generation
  - `evaluate.py` --- VLM API dispatch (Anthropic, OpenAI) with JSON/freetext response parsing
  - `storage.py` --- CSV append/load via polars
  - `analysis.py` --- accuracy breakdowns by model, layout, side, colour pair
  - `cli.py` --- typer CLI with `generate`, `evaluate`, `analyse` subcommands
- `tests/` --- pytest tests

## Results CSV schema

Each row is one trial: `model`, `crisp_on_top`, `crisp_side`, `colour_crisp`, `colour_blurred`, `correct_answer`, `parsed_answer`, `correct`, `prompt`, `raw_response`, `timestamp`.

## Conventions

- use `uv` via mise for all Python tooling
- polars for dataframes, pydantic for validation
- Australian English in prose (colour, analyse, etc.)
- `stimuli/` and `results/` are gitignored
