# vlm-perception

Do vision-language models (VLMs) assume that crisp objects are in front of
blurred objects?

This project generates simple stimuli --- pairs of overlapping circles where one
is crisp and one is Gaussian-blurred --- and asks VLMs to determine which circle
occludes the other. The hypothesis is that current VLMs struggle when the
blurred circle is in front, because they have a prior that crisp means
foreground.

## Setup

Requires [mise](https://mise.jdx.dev/) (which provides `uv`):

```sh
mise install
uv sync
```

Set API keys for the providers you want to test:

```sh
export ANTHROPIC_API_KEY="..."
export OPENAI_API_KEY="..."
```

## Usage

### Generate stimuli

```sh
uv run vlm-perception generate                # full factorial (120 images, blur=20px)
uv run vlm-perception generate --blur-sweep   # blur sweep (96 images, 6 blur levels)
```

The full factorial produces 120 images at the default blur radius (20px). The
blur sweep produces 96 images across 6 blur levels (0, 4, 8, 12, 16, 20px)
with a reduced set of 4 colour pairs. The 0px level (no blur) serves as a
baseline where models have no blur cue.

### Run evaluation

```sh
uv run vlm-perception evaluate --model claude-sonnet-4-6 --reps 3
uv run vlm-perception evaluate --model gpt-5.4-mini --reps 3 --prompt minimal
uv run vlm-perception evaluate --model claude-sonnet-4-6 --reps 3 --prompt thinking
```

Multiple models and prompts can be specified in a single run --- they execute
concurrently via asyncio with per-provider rate limiting:

```sh
uv run vlm-perception evaluate \
  --model claude-sonnet-4-6 --model gpt-5.4-mini \
  --prompt neutral --prompt cot \
  --reps 3
```

Use `--blur-sweep` to evaluate the reduced blur radius sweep conditions (96)
instead of the full factorial (120):

```sh
uv run vlm-perception evaluate --model claude-sonnet-4-6 --reps 3 --blur-sweep
```

Available models: `claude-opus-4-6`, `claude-sonnet-4-6`, `claude-haiku-4-5`,
`gpt-5.4`, `gpt-5.4-mini`, `gpt-5.4-nano`. Use `--limit N` to evaluate only the
first N conditions. Use `--prompt <id>` to select a prompt variant (default:
`neutral`). Use `--concurrency N` to set max concurrent requests per provider
(default: 10). Use `--resume` to skip trials already present in the results file
--- useful for recovering from partial runs or errors.

### Prompt variants

- **neutral** (default) --- describes the image and asks which circle is in front
- **minimal** --- bare question with no framing
- **foreground** --- uses explicit foreground/background terminology
- **psychophysics** --- experimental framing mentioning sharpness and blur
- **cot** --- chain-of-thought: asks the model to reason step by step about edge
  continuity in the overlap region before answering. The MAD'26 paper labels
  this "Scripted CoT" to distinguish prescribed-step prompting from the
  free-form reasoning that the `thinking` variant enables.
- **thinking** --- same text as `neutral` but enables provider-level reasoning
  tokens (Anthropic extended thinking / OpenAI `reasoning_effort="medium"`)

Prompt definitions are in `src/vlm_perception/prompts.json`.

Results are appended to `results/results.jsonl`.

### Plot figures

```sh
uv run vlm-perception plot --results-path results/results.jsonl
```

Generates dose-response and prompt invariance charts as PDF.

### Analyse results

```sh
uv run vlm-perception analyse
```

Prints a full statistical report: depth order effect (Fisher exact, odds ratios),
blur dose-response (Cochran-Armitage trend), zero-blur baseline (binomial test),
model and prompt effects (chi-square with Holm-corrected pairwise comparisons),
and a summary table.

### Judge reasoning traces

```sh
uv run vlm-perception judge --concurrency 12
```

Uses Claude Sonnet 4.6 as an automated judge to label each reasoning trace
(from `cot` and `thinking` prompts) on bias-incongruent trials along seven
boolean dimensions related to the verbal heuristic ("sharp = closer")
hypothesis. See `src/vlm_perception/judge.py` for the rubric and the
self-judgment-bias caveat. Judgments append to `results/judgments.jsonl`.

By default only bias-incongruent trials are judged (where the heuristic drives
errors). Use `--include-bias-congruent` to judge both conditions, `--limit N`
to test on a subset first.

Aggregate the labels into per-model frequency tables:

```sh
uv run vlm-perception analyse-judgments
```

Outputs separate breakdowns for the `cot` and `thinking` prompts.

## Experimental design

### Stimulus parameters

- canvas: 512x512px, background RGB (128, 128, 128)
- circle radius: 100px, centre offset: 75px (~25% area overlap)
- colours: 6 OKLCH hues at L=0.7, C=0.15 (red, yellow, green, cyan, blue,
  magenta)

### Full factorial (120 conditions)

The original design fully crosses depth order, spatial position, and colour
pairs at a fixed blur radius of 20px:

- **depth order** (2): crisp on top, blurred on top
- **spatial position** (2): crisp circle on left, crisp circle on right
- **colour pairs** (30): 6 x 5 hue combinations, excluding same-colour

### Blur radius sweep (96 conditions)

A preliminary full-factorial study showed no significant effects of colour pair
or spatial position. The blur sweep therefore uses a reduced design to
efficiently test the effect of blur strength:

- **blur radius** (6): 0, 4, 8, 12, 16, 20px
- **depth order** (2): crisp on top, blurred on top
- **spatial position** (2): crisp circle on left, crisp circle on right
- **colour pairs** (4): red/cyan, yellow/blue, green/magenta, cyan/red ---
  complementary pairs spanning the hue wheel

### Dependent variable

Binary left/right response parsed from VLM output. When using the `thinking`
prompt variant, reasoning traces (Anthropic extended thinking) are also
captured.

## Licence

MIT --- see [LICENSE](LICENSE).

Copyright (c) 2026 Ben Swift, Jess Herrington
