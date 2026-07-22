---
id: TASK-3
title: test the Scotoma text variant against real VLMs
status: To Do
assignee: []
created_date: '2026-07-22 06:28'
updated_date: '2026-07-22 07:07'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Extend the occlusion-edge-blur experiment from circles to the Scotoma text typeface (src/vlm_perception/scotoma/), measuring whether VLMs read the crisp "robot" stream instead of the blurred "real" stream a human reads. The rendering side already exists; this task is the experimental harness around it.

Design (settled 2026-07-22):

1. String pool. Space-free, length-matched uppercase strings: 8 English words of 10 letters (roughly frequency-matched), plus a matched pseudoword set of the same size and length to rule out the language prior. Space-free is a hard constraint: in pair_aligned, a space in one stream renders the other stream glyph solo --- crisp and unoccluded --- leaking that stream for free. Enforce a minimum pairwise positional Hamming distance within each set (>= 8 of 10 positions differing) so the bias index is not attenuated by similar pairs, and record the pair distance d(A,B) per trial as a covariate anyway.

2. Pairing and counterbalancing. Partition each 8-string set into 4 disjoint pairs; render each pair in both role orderings (the reciprocal diptych logic) so every string appears once as real and once as robot: 8 ordered pairs per set, 16 total. Counterbalance colour-role assignment across ordered pairs (half red-real/cyan-robot, half swapped) rather than crossing it as a factor --- this removes the blur-hue confound at zero extra trials. Note ScotomaStyle currently hard-codes colour_real=red / colour_robot=cyan; the generator must set both assignments.

3. Factors. Blur (6 levels: 0 plus 5 nonzero blur_fraction values), depth order (2: blurred-on-top exploit, crisp-on-top congruent control), 16 ordered pairs. Offset fixed at the 0.38 default for the headline sweep; an offset sub-sweep at one or two blur levels is an optional follow-up, not part of this task. Condition count: 16 x 6 x 2 = 192.

4. Trial budget. 192 conditions x 3 reps x 4 prompts (naive, dual-stream, cot, thinking) x 6 models = 13,824 trials, plus ~288 legibility-baseline trials --- about 1.4x the 10,368-trial circle blur sweep, roughly 20 minutes at concurrency 10.

5. Dependent variable. Normalise the model output (uppercase, strip non-letter characters) and score it against BOTH streams with two metrics: normalised Levenshtein (robust to dropped/inserted characters) and positional Hamming (physically meaningful, since glyph cells align position-by-position). Compute the bias index (d_real - d_robot) / (d_real + d_robot) in [-1, 1] for each metric; it degrades gracefully towards 0 when the model reads mush (both distances high). This is the biggest new piece --- the left/right parsing in evaluate.py does not apply.

6. Legibility baseline. Render each string solo (unblurred, no overlay) and confirm each model transcribes plain Jost caps correctly first. Without this a null result is uninterpretable --- we cannot tell "resisted the illusion" from "cannot read the font at this resolution".

7. Resolution pre-check. Before the main sweep, verify the chosen font/canvas size survives provider-side downsampling (Anthropic and OpenAI both resize/tile images) by running the solo-string legibility check at 2-3 candidate sizes and picking the smallest that transcribes cleanly.

8. Prompt design. Naive "what does this say?" is the headline measure (it captures default reading). Add an explicit "there may be two overlapping messages, transcribe both" prompt, plus cot and thinking variants mirroring prompts.json. Telling the model there are two streams changes the task, so the naive prompt stays primary.

9. Storage + analysis. New JSONL schema (string_real, string_robot, blur_px, offset_fraction, blurred_on_top, colour_real, model, prompt_id, raw_transcription, dist_real_lev, dist_robot_lev, dist_real_ham, dist_robot_ham, bias_index_lev, bias_index_ham, d_pair, reasoning_trace, timestamp) in its own results file, and a new analysis path kept separate from the circle analysis: bias index vs blur (dose-response), depth-order effect, English vs pseudoword contrast, per model. The fixed balanced pool supports all of these because string pair is a nuisance factor --- only per-string effects are unestimable, and we do not need them.

10. Optional but strengthening: a small human-transcription check (n approx 5) on a handful of stimuli, to ground the "humans read the blurred stream" half of the claim, which is currently asserted from vision-science theory alone.

Context: blog post at benswift.me, "A typeface for humans, not machines" (https://benswift.me/blog/2026/07/22/a-typeface-for-humans-not-machines/), and the MAD 26 paper it builds on (https://doi.org/10.1145/3810988.3812661).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A per-phrase unblurred legibility baseline is captured per model, so null results are interpretable
- [ ] #2 A random-string / pseudoword control condition is included to rule out the language prior
- [ ] #3 A stimulus generator produces the Scotoma condition set from a space-free, length-matched string pool with a minimum pairwise Hamming distance, with colour-role assignment and role order counterbalanced, blur swept, offset fixed at 0.38, and both depth orders rendered
- [ ] #4 A transcription evaluate path normalises model output and scores each trial against both streams (normalised Levenshtein and positional Hamming, bias index for each, pair distance d(A,B) recorded) and appends to a dedicated JSONL results file
- [ ] #5 A resolution pre-check confirms the chosen font/canvas size survives provider image downsampling before the main sweep is run
- [ ] #6 Analysis reports bias index vs blur radius (dose-response), the depth-order effect, and the English vs pseudoword contrast, per model
- [ ] #7 The Scotoma evaluate path appends each trial as it completes and supports --resume, so an interrupted run (rate limits, credit exhaustion) loses no collected data and can be continued in place
<!-- AC:END -->
