# vlm-perception

VLM perception experiment: testing whether vision-language models can correctly
identify occlusion order of crisp vs blurred overlapping circles.

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

### Blur radius sweep (96 conditions)

Reduced design motivated by preliminary full-factorial results showing no
significant effects of colour pair or spatial position:

- **blur radius** (6): 0, 4, 8, 12, 16, 20px (0 = no blur baseline)
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

## Scotoma transcription experiment

Tests whether VLMs transcribe the crisp "robot" stream of a Scotoma render
rather than the blurred "real" stream a human reads. 192 conditions: 16 ordered
string pairs (8 English 10-letter words + 8 matched pseudowords, pairwise
positional Hamming >= 8 within each set, colour-role and role order
counterbalanced) x 6 blur fractions (0, 0.02, 0.04, 0.07, 0.10, 0.14) x 2 depth
orders, offset fixed at 0.38, font size 96 (validated by `scotoma precheck`
against provider downsampling).

```sh
uv run vlm-perception scotoma generate            # 192 main + 16 solo stimuli
uv run vlm-perception scotoma evaluate --model claude-sonnet-4-6 \
  --prompt naive --reps 3 --resume                # main sweep (per-trial append)
uv run vlm-perception scotoma evaluate --model claude-sonnet-4-6 \
  --legibility --reps 3 --resume                  # solo-string baseline
uv run vlm-perception scotoma analyse             # bias-index report
```

Results append per-trial to `results/scotoma.jsonl` (schema: `string_real`,
`string_robot`, `blur_fraction`, `blurred_on_top`, `colour_real`,
`raw_transcription`, `dist_{real,robot}_{lev,ham}`, `bias_index_{lev,ham}`,
`d_pair`, ...). Bias index convention: +1 = read the crisp robot stream (the
exploit works), -1 = read the blurred real stream (human-like reading), ~0 =
mush or both streams recovered.

## Conventions

- `stimuli/` is gitignored; `results/` is tracked
