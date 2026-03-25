# vlm-perception

VLM perception experiment: testing whether vision-language models can correctly
identify occlusion order of crisp vs blurred overlapping circles.

## Quick start

```sh
mise install          # install uv
uv sync               # install Python dependencies
uv run vlm-perception generate  # generate 120 stimulus images in stimuli/
```

To run VLM evaluation, set the relevant API key and run:

```sh
export ANTHROPIC_API_KEY="..."
uv run vlm-perception evaluate --model claude-sonnet-4-6 --reps 3
uv run vlm-perception evaluate --model claude-sonnet-4-6 --reps 3 --prompt minimal

export OPENAI_API_KEY="..."
uv run vlm-perception evaluate --model gpt-5.4-mini --reps 3
```

Available models: `claude-opus-4-6`, `claude-sonnet-4-6`, `claude-haiku-4-5`,
`gpt-5.4`, `gpt-5.4-mini`, `gpt-5.4-nano`. Use `--limit N` to evaluate only the
first N conditions. Use `--prompt <id>` to select a prompt variant (default:
`neutral`). Available prompts: `neutral`, `minimal`, `foreground`,
`psychophysics`. Prompt definitions are in `src/vlm_perception/prompts.json`.

Approximate evaluation time for 360 trials (120 conditions x 3 reps), median per
trial:

| Model             | Per trial | 360 trials |
| ----------------- | --------- | ---------- |
| claude-opus-4-6   | ~2.4s     | ~15min     |
| claude-sonnet-4-6 | ~1.2s     | ~7min      |
| claude-haiku-4-5  | ~1.3s     | ~8min      |
| gpt-5.4           | ~1.0s     | ~6min      |
| gpt-5.4-mini      | ~0.9s     | ~5min      |
| gpt-5.4-nano      | ~0.7s     | ~4min      |

Results append to `results/results.csv`. Analyse with:

```sh
uv run vlm-perception analyse
```

Run tests with:

```sh
uv run pytest -x
```

## Experimental design

Each stimulus image shows two overlapping circles on a mid-grey background. One
circle is crisp, the other is Gaussian-blurred. The question posed to the VLM
is: which circle is in front (occluding the other)?

### Factorial conditions (120 total)

- **depth order** (2): crisp on top, blurred on top
- **spatial position** (2): crisp circle on left, crisp circle on right
- **colour pairs** (6 x 5 = 30): 6 hues (OKLCH, perceptually uniform) for each
  circle, excluding same-colour pairs

### Stimulus parameters

- canvas: 512x512px, background RGB (128, 128, 128)
- circle radius: 100px, centre offset: 75px (~25% area overlap)
- blur: Gaussian radius 20px
- colours: 6 OKLCH hues at L=0.7, C=0.15 (red, yellow, green, cyan, blue,
  magenta)

## Project structure

- `src/vlm_perception/` --- main package
  - `models.py` --- pydantic models (Colour, Side, Condition, TrialResult),
    `MODEL_REGISTRY` for supported VLMs, and `all_conditions()` factorial
    generator
  - `stimuli.py` --- Pillow-based stimulus image generation
  - `prompts.json` --- prompt registry mapping IDs to full prompt text
  - `evaluate.py` --- VLM API dispatch (Anthropic, OpenAI) with JSON/freetext
    response parsing
  - `storage.py` --- CSV append/load via polars
  - `analysis.py` --- accuracy breakdowns by model, layout, side, colour pair
  - `cli.py` --- typer CLI with `generate`, `evaluate`, `analyse` subcommands
- `tests/` --- pytest tests

## Results CSV schema

Each row is one trial: `model`, `prompt_id`, `crisp_on_top`, `crisp_side`,
`colour_crisp`, `colour_blurred`, `correct_answer`, `parsed_answer`, `correct`,
`prompt`, `raw_response`, `timestamp`.

## Conventions

- use `uv` via mise for all Python tooling
- polars for dataframes, pydantic for validation
- Australian English in prose (colour, analyse, etc.)
- `stimuli/` is gitignored; `results/` is tracked
