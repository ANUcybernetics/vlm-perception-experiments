---
id: TASK-3
title: test the Scotoma text variant against real VLMs
status: To Do
assignee: []
created_date: '2026-07-22 06:28'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Extend the occlusion-edge-blur experiment from circles to the Scotoma text
typeface (`src/vlm_perception/scotoma/`), measuring whether VLMs read the crisp
"robot" stream instead of the blurred "real" stream a human reads. The rendering
side already exists; this task is the experimental harness around it.

Ben's starting thoughts:

- colour probably doesn't need varying (no effect in the circle study) --- fix
  one pair (red/cyan).
- blur radius is the primary factor, and offset may matter too.
- vary the text: pick a small pool of phrases and run all combinations of them
  in both the human (blurred/front) and robot (crisp/behind) positions. This
  means no full Cartesian product over letters; counterbalance instead.

New machinery needed beyond reusing the async dispatch:

1. Dependent variable. The circle task was binary; this one is transcription.
   Score the model's output by normalised Levenshtein distance to BOTH streams
   and compute a bias index `(d_real - d_robot) / (d_real + d_robot)` in [-1,
   1]. It degrades gracefully when the model reads mush (both distances high,
   index near 0). This is the biggest new piece --- `evaluate.py`'s left/right
   parsing does not apply.

2. Length-matched phrase pool. `pair_aligned` requires equal-length strings, so
   the pool should be a set of same-length phrases (e.g. all 15 characters) so
   any phrase can pair with any other. Ideally frequency-matched English, plus a
   random-string / pseudoword control set to rule out the language prior doing
   the work rather than the visual cue.

3. Legibility baseline. Render each phrase solo (unblurred, no overlay) and
   confirm each model transcribes plain Jost caps correctly first. Without this
   a null result is uninterpretable --- we can't tell "resisted the illusion"
   from "can't read the font at this resolution".

4. Prompt design. Naive "what does this say?" is the clean condition (it
   measures default reading). Add an explicit "there may be two overlapping
   messages, transcribe both" prompt, plus CoT and thinking variants mirroring
   `prompts.json`. Note that telling the model there are two streams changes the
   task, so keep the naive prompt as the headline measure.

5. Depth-order control. Render both blurred-on-top (the exploit) and
   crisp-on-top (the congruent control, where the crisp stream genuinely is in
   front). blur = 0 is the other key control (both streams crisp, no depth cue).

6. Position counterbalancing. Render each phrase pair both ways (A-real/B-robot
   and B-real/A-robot --- the reciprocal diptych) so any phrase-intrinsic
   legibility difference cancels out.

7. Image resolution. VLMs downsample images before their vision encoder, so fix
   a font/canvas size large enough that the fragmented glyphs survive, and
   sanity-check a couple of sizes rather than assuming.

8. Separate storage + analysis. New JSONL schema (`phrase_real`, `phrase_robot`,
   `blur_px`, `offset_fraction`, `blurred_on_top`, `model`, `prompt_id`,
   `raw_transcription`, `dist_real`, `dist_robot`, `bias_index`,
   `reasoning_trace`, `timestamp`) in its own results file, and a new analysis
   path (bias-index dose-response over blur, depth-order effect, per-model),
   kept separate from the circle analysis.

9. Optional but strengthening: a small human-transcription check (n
   approximately 5) on a handful of stimuli, to ground the "humans read the
   blurred stream" half of the claim, which is currently asserted from
   vision-science theory alone.

Context: blog post at benswift.me, "A typeface for humans, not machines"
(https://benswift.me/blog/2026/07/22/a-typeface-for-humans-not-machines/), and
the MAD'26 paper it builds on (https://doi.org/10.1145/3810988.3812661).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A stimulus generator produces the Scotoma condition set from a length-matched phrase pool (colour fixed, blur and offset swept, both depth orders, positions counterbalanced)
- [ ] #2 A transcription evaluate path scores each trial against both streams (normalised Levenshtein + bias index) and appends to a dedicated JSONL results file
- [ ] #3 A per-phrase unblurred legibility baseline is captured per model, so null results are interpretable
- [ ] #4 A random-string / pseudoword control condition is included to rule out the language prior
- [ ] #5 Analysis reports bias index vs blur radius (dose-response) and the depth-order effect, per model
<!-- AC:END -->
