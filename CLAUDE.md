# vlm-perception

VLM perception experiment: testing whether vision-language models can correctly
identify occlusion order of crisp vs blurred overlapping circles.

## Quick start

```sh
mise install          # install uv
uv sync               # install Python dependencies
uv run vlm-perception generate              # full factorial (120 images)
uv run vlm-perception generate --blur-sweep # blur sweep (80 images, 5 blur levels)
```

To run VLM evaluation, set the relevant API key and run:

```sh
export ANTHROPIC_API_KEY="..."
export OPENAI_API_KEY="..."
uv run vlm-perception evaluate --model claude-sonnet-4-6 --reps 3
uv run vlm-perception evaluate --model claude-sonnet-4-6 --reps 3 --prompt minimal
uv run vlm-perception evaluate --model gpt-5.4-mini --reps 3
```

Multiple models and/or prompts can be specified in a single run --- they execute
concurrently via asyncio with per-provider rate limiting:

```sh
uv run vlm-perception evaluate \
  --model claude-sonnet-4-6 --model gpt-5.4-mini \
  --prompt neutral --prompt cot \
  --reps 3
```

Available models: `claude-opus-4-6`, `claude-sonnet-4-6`, `claude-haiku-4-5`,
`gpt-5.4`, `gpt-5.4-mini`, `gpt-5.4-nano`. Use `--limit N` to evaluate only the
first N conditions. Use `--prompt <id>` to select a prompt variant (default:
`neutral`). Use `--concurrency N` to set max concurrent requests per provider
(default: 10). Use `--resume` to skip already-completed trials (based on
existing results file). Use `--blur-sweep` to evaluate the reduced blur sweep conditions
(80) instead of the full factorial (120). Available prompts: `neutral`,
`minimal`, `foreground`, `psychophysics`, `cot`, `thinking`. Prompt definitions
are in `src/vlm_perception/prompts.json`.

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

Results append to `results/results.jsonl`. Analyse with:

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

### Stimulus parameters

- canvas: 512x512px, background RGB (128, 128, 128)
- circle radius: 100px, centre offset: 75px (~25% area overlap)
- colours: 6 OKLCH hues at L=0.7, C=0.15 (red, yellow, green, cyan, blue,
  magenta)

### Full factorial conditions (120 total)

- **depth order** (2): crisp on top, blurred on top
- **spatial position** (2): crisp circle on left, crisp circle on right
- **colour pairs** (6 x 5 = 30): 6 hues, excluding same-colour pairs
- **blur radius**: fixed at 20px

### Blur radius sweep (80 conditions)

Reduced design motivated by preliminary full-factorial results showing no
significant effects of colour pair or spatial position:

- **blur radius** (5): 4, 8, 12, 16, 20px
- **depth order** (2): crisp on top, blurred on top
- **spatial position** (2): crisp circle on left, crisp circle on right
- **colour pairs** (4): red/cyan, yellow/blue, green/magenta, cyan/red

### Prompt variants

- **neutral** --- describes the image and asks which circle is in front
- **minimal** --- bare question with no framing
- **foreground** --- uses explicit foreground/background terminology
- **psychophysics** --- experimental framing mentioning sharpness and blur
- **cot** --- chain-of-thought: asks the model to reason step by step about edge
  continuity in the overlap region before answering
- **thinking** --- uses the same text as `neutral` but enables provider-level
  reasoning tokens (Anthropic extended thinking / OpenAI `reasoning_effort`).
  For Anthropic, Opus uses adaptive thinking; Sonnet and Haiku use a 4096-token
  budget. For OpenAI, `reasoning_effort="medium"` is set.

## Project structure

- `src/vlm_perception/` --- main package
  - `models.py` --- pydantic models (Colour, Side, Condition, TrialResult),
    `MODEL_REGISTRY` for supported VLMs, `all_conditions()` factorial generator,
    and `blur_sweep_conditions()` for the reduced blur sweep design
  - `stimuli.py` --- Pillow-based stimulus image generation
  - `prompts.json` --- prompt registry mapping IDs to full prompt text
  - `evaluate.py` --- sync and async VLM API dispatch (Anthropic, OpenAI) with
    JSON/freetext response parsing and per-provider semaphore rate limiting
  - `storage.py` --- JSONL append/load via polars, with async support
  - `analysis.py` --- accuracy breakdowns by model, layout, side, blur radius,
    colour pair
  - `cli.py` --- typer CLI with `generate`, `evaluate`, `analyse` subcommands
- `tests/` --- pytest tests

## Results JSONL schema

Each line is a JSON object with fields: `model`, `prompt_id`, `blur_px`,
`crisp_on_top`, `crisp_side`, `colour_crisp`, `colour_blurred`,
`correct_answer`, `parsed_answer`, `correct`, `prompt`, `raw_response`,
`reasoning_trace`, `timestamp`.

## Conventions

- use `uv` via mise for all Python tooling
- polars for dataframes, pydantic for validation
- Australian English in prose (colour, analyse, etc.)
- `stimuli/` is gitignored; `results/` is tracked
