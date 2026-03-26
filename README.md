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
uv run vlm-perception generate
```

Produces 120 images (2 depth orders x 2 spatial positions x 6 x 5 colour pairs,
excluding same-colour) in `stimuli/`.

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

Available models: `claude-opus-4-6`, `claude-sonnet-4-6`, `claude-haiku-4-5`,
`gpt-5.4`, `gpt-5.4-mini`, `gpt-5.4-nano`. Use `--limit N` to evaluate only the
first N conditions. Use `--prompt <id>` to select a prompt variant (default:
`neutral`). Use `--concurrency N` to set the max concurrent requests per provider
(default: 10).

### Prompt variants

- **neutral** (default) --- describes the image and asks which circle is in front
- **minimal** --- bare question with no framing
- **foreground** --- uses explicit foreground/background terminology
- **psychophysics** --- experimental framing mentioning sharpness and blur
- **cot** --- chain-of-thought: asks the model to reason step by step about edge
  continuity in the overlap region before answering
- **thinking** --- same text as `neutral` but enables provider-level reasoning
  tokens (Anthropic extended thinking / OpenAI `reasoning_effort="medium"`)

Prompt definitions are in `src/vlm_perception/prompts.json`.

Results are appended to `results/results.jsonl`.

### Analyse results

```sh
uv run vlm-perception analyse
```

Prints accuracy breakdowns by model, layout (crisp-on-top vs blurred-on-top),
spatial position, and colour pair.

## Experimental design

- **Stimuli**: two overlapping circles on a mid-grey (128, 128, 128) background,
  one crisp and one Gaussian-blurred (radius 20px). Circles are 100px radius with
  75px horizontal offset between centres (~25% overlap).
- **Conditions**: 2 (depth order) x 2 (crisp on left/right) x 6 x 5 (colour
  pairs from 6 equidistant hues, excluding same-colour) = 120 conditions.
- **Colours**: red, yellow, green, cyan, blue, magenta --- 6 equally-spaced hues
  in OKLCH (L=0.7, C=0.15) for perceptual uniformity.
- **Dependent variable**: binary left/right response parsed from VLM output.

## Licence

MIT --- see [LICENSE](LICENSE).

Copyright (c) 2026 Ben Swift, Jess Herrington
