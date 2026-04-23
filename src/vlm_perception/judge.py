"""LLM-as-judge categorisation of MLLM reasoning traces.

For each reasoning trace produced by a tested model on a bias-incongruent
trial, ask Claude Sonnet 4.6 to label the trace along seven boolean
dimensions:

- articulates_sharp_closer: explicitly states or implies "sharp = closer"
- articulates_blur_far: explicitly states or implies "blur = further"
- invokes_dof: mentions depth of field, bokeh, defocus, or photography rule
- attempts_occlusion_reasoning: tries to reason from edge continuity / overlap
- identifies_correct_occluder: at any point names the actually-in-front circle
- self_corrects: surfaces contradictory evidence to its initial impression
- reverts_to_bias: if it self-corrects, ends with the bias-congruent answer

Reasoning text is taken from `reasoning_trace` for the *thinking* prompt and
from `raw_response` for the *cot* prompt (where the chain-of-thought is part
of the visible response, with the JSON answer at the end).

Limitation: a single judge (Claude Sonnet 4.6) is used for all traces,
including those produced by Sonnet itself and other Anthropic models.
Self-judgment bias may inflate agreement on heuristic-articulation labels for
Anthropic-generated traces. We accept this trade-off for consistency and
cost; readers should treat per-provider comparisons of the
`articulates_sharp_closer` and `articulates_blur_far` labels as
suggestive rather than definitive.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anthropic
import polars as pl
from pydantic import BaseModel, Field

from vlm_perception.storage import load_results

log = logging.getLogger(__name__)

JUDGE_MODEL = "claude-sonnet-4-6"
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0


JUDGE_SYSTEM_PROMPT = """You are evaluating reasoning traces produced by multimodal language models attempting a depth-perception task. Each trace is the model's verbatim reasoning when shown two overlapping circles (one with crisp edges, one with blurred edges) and asked which circle is in front (occluding the other).

In every trace you will judge, the BLURRED circle is actually in front (occluding the crisp one). The "correct" answer is therefore the blurred circle. The "bias-congruent" answer (which is wrong on this trial) would be to say the crisp circle is in front.

Your job is to categorise the trace along seven boolean dimensions. Read carefully and answer each independently.

LABEL DEFINITIONS:

1. articulates_sharp_closer (bool)
   TRUE if the trace explicitly states or strongly implies that sharp/crisp/well-defined edges indicate the object is closer, in front, in the foreground, or nearer to the viewer. Examples that count:
   - "the sharp edges suggest the circle is in front"
   - "crisp boundaries typically appear closer"
   - "the well-defined edge means it's the foreground object"
   FALSE if the trace merely describes one circle as "sharp" without linking sharpness to depth/proximity.

2. articulates_blur_far (bool)
   TRUE if the trace explicitly states or strongly implies that blurred/diffuse/soft edges indicate the object is further away, behind, in the background, or distant. Examples that count:
   - "the blurred edges suggest distance"
   - "the soft outline indicates the circle is behind"
   - "blur means the object is in the background"
   FALSE if the trace merely describes one circle as "blurred" without linking blur to depth.

3. invokes_dof (bool)
   TRUE if the trace explicitly mentions depth of field, defocus, bokeh, focus distance, or any photography-rule framing (e.g. "objects in focus are closer to the focal plane", "depth-of-field blur", "out-of-focus elements"). FALSE otherwise.

4. attempts_occlusion_reasoning (bool)
   TRUE if the trace tries to reason about edge continuity, overlap geometry, or which circle's boundary "continues through" or is "interrupted" at the overlap region (whether it gets the right answer or not). FALSE if the trace never engages with overlap geometry at all.

5. identifies_correct_occluder (bool)
   TRUE if at ANY point in the trace the model correctly identifies the BLURRED circle as the one in front / occluding / overlapping the crisp one (even if it later changes its mind). FALSE if the trace never reaches this conclusion.

6. self_corrects (bool)
   TRUE if the trace contains a moment where the model surfaces evidence that contradicts its initial reading and explicitly notes the conflict. Markers include "but wait", "actually", "on closer inspection", "let me reconsider", "however, looking again", or any clear pivot from one tentative answer to reconsidering. FALSE if the reasoning proceeds linearly without backtracking.

7. reverts_to_bias (bool)
   TRUE *only if* self_corrects is also TRUE *and* the trace ultimately concludes with the bias-congruent answer (crisp = in front) despite the contradictory evidence it surfaced. FALSE if either the trace doesn't self-correct, or it self-corrects and sticks with the corrected (correct) answer.

OUTPUT FORMAT:

Use the `record_judgment` tool with one boolean for each label. Do not output any other text.
"""


JUDGE_TOOL: dict[str, Any] = {
    "name": "record_judgment",
    "description": "Record the seven boolean labels for the trace.",
    "input_schema": {
        "type": "object",
        "properties": {
            "articulates_sharp_closer": {"type": "boolean"},
            "articulates_blur_far": {"type": "boolean"},
            "invokes_dof": {"type": "boolean"},
            "attempts_occlusion_reasoning": {"type": "boolean"},
            "identifies_correct_occluder": {"type": "boolean"},
            "self_corrects": {"type": "boolean"},
            "reverts_to_bias": {"type": "boolean"},
        },
        "required": [
            "articulates_sharp_closer",
            "articulates_blur_far",
            "invokes_dof",
            "attempts_occlusion_reasoning",
            "identifies_correct_occluder",
            "self_corrects",
            "reverts_to_bias",
        ],
    },
}


class Judgment(BaseModel):
    articulates_sharp_closer: bool
    articulates_blur_far: bool
    invokes_dof: bool
    attempts_occlusion_reasoning: bool
    identifies_correct_occluder: bool
    self_corrects: bool
    reverts_to_bias: bool


@dataclass
class JudgeRecord:
    """Identifies the source trial and stores the trace + judgment."""

    model: str  # the model that produced the trace
    prompt_id: str  # cot or thinking
    blur_px: int
    crisp_on_top: bool
    colour_crisp: str
    colour_blurred: str
    trace_source_field: str  # "reasoning_trace" or "raw_response"
    trace_text: str
    judgment: Judgment | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "model": self.model,
            "prompt_id": self.prompt_id,
            "blur_px": self.blur_px,
            "crisp_on_top": self.crisp_on_top,
            "colour_crisp": self.colour_crisp,
            "colour_blurred": self.colour_blurred,
            "trace_source_field": self.trace_source_field,
            "trace_text": self.trace_text,
            "error": self.error,
        }
        if self.judgment is not None:
            d["judgment"] = self.judgment.model_dump()
        else:
            d["judgment"] = None
        return d


def extract_trace(row: dict[str, Any]) -> tuple[str, str] | None:
    """Return (text, source_field) or None if no trace is available."""
    prompt_id = row.get("prompt_id")
    if prompt_id == "thinking":
        text = (row.get("reasoning_trace") or "").strip()
        if text:
            return text, "reasoning_trace"
    elif prompt_id == "cot":
        text = (row.get("raw_response") or "").strip()
        if text:
            return text, "raw_response"
    return None


def build_records(df: pl.DataFrame, *, only_bias_incongruent: bool = True) -> list[JudgeRecord]:
    rows = df.to_dicts()
    records: list[JudgeRecord] = []
    for row in rows:
        if row.get("prompt_id") not in ("cot", "thinking"):
            continue
        if only_bias_incongruent and row.get("crisp_on_top"):
            continue
        extracted = extract_trace(row)
        if extracted is None:
            continue
        text, source = extracted
        records.append(
            JudgeRecord(
                model=row["model"],
                prompt_id=row["prompt_id"],
                blur_px=int(row["blur_px"]),
                crisp_on_top=bool(row["crisp_on_top"]),
                colour_crisp=row["colour_crisp"],
                colour_blurred=row["colour_blurred"],
                trace_source_field=source,
                trace_text=text,
            )
        )
    return records


async def judge_one(
    client: anthropic.AsyncAnthropic,
    record: JudgeRecord,
    semaphore: asyncio.Semaphore,
) -> JudgeRecord:
    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                response = await client.messages.create(
                    model=JUDGE_MODEL,
                    max_tokens=512,
                    system=[
                        {
                            "type": "text",
                            "text": JUDGE_SYSTEM_PROMPT,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    tools=[JUDGE_TOOL],
                    tool_choice={"type": "tool", "name": "record_judgment"},
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                f"Trace produced by {record.model} "
                                f"under the {record.prompt_id} prompt. "
                                f"On this trial the BLURRED circle is in front "
                                f"(blur radius {record.blur_px} px).\n\n"
                                f"--- TRACE BEGINS ---\n{record.trace_text}\n"
                                f"--- TRACE ENDS ---"
                            ),
                        }
                    ],
                )
                break
            except (
                anthropic.InternalServerError,
                anthropic.APIConnectionError,
                anthropic.RateLimitError,
            ) as exc:
                if attempt == MAX_RETRIES - 1:
                    record.error = f"{type(exc).__name__}: {exc}"
                    return record
                await asyncio.sleep(RETRY_BASE_DELAY * (2**attempt))
        else:
            record.error = "max retries exhausted"
            return record

        for block in response.content:
            if block.type == "tool_use" and block.name == "record_judgment":
                try:
                    record.judgment = Judgment.model_validate(block.input)
                except Exception as exc:  # noqa: BLE001
                    record.error = f"validation: {exc}"
                return record
        record.error = "no tool_use block in response"
        return record


async def judge_all(
    records: list[JudgeRecord],
    *,
    concurrency: int = 8,
    output_path: Path | None = None,
) -> list[JudgeRecord]:
    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [judge_one(client, r, semaphore) for r in records]

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        completed: list[JudgeRecord] = []
        with output_path.open("w") as f:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                completed.append(result)
                f.write(json.dumps(result.to_dict()) + "\n")
                f.flush()
                if len(completed) % 100 == 0:
                    log.info("judged %d / %d", len(completed), len(records))
        return completed

    return await asyncio.gather(*tasks)


def run_judge(
    results_path: Path,
    output_path: Path,
    *,
    limit: int | None = None,
    concurrency: int = 8,
    only_bias_incongruent: bool = True,
) -> None:
    df = load_results(results_path)
    records = build_records(df, only_bias_incongruent=only_bias_incongruent)
    if limit is not None:
        records = records[:limit]
    log.info("judging %d traces with %s", len(records), JUDGE_MODEL)
    asyncio.run(
        judge_all(records, concurrency=concurrency, output_path=output_path)
    )
    log.info("wrote %s", output_path)


# --- aggregation helpers ---


LABEL_FIELDS = (
    "articulates_sharp_closer",
    "articulates_blur_far",
    "invokes_dof",
    "attempts_occlusion_reasoning",
    "identifies_correct_occluder",
    "self_corrects",
    "reverts_to_bias",
)


def load_judgments(judgments_path: Path) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    with judgments_path.open() as f:
        for line in f:
            d = json.loads(line)
            j = d.get("judgment") or {}
            row = {
                "model": d["model"],
                "prompt_id": d["prompt_id"],
                "blur_px": d["blur_px"],
                "crisp_on_top": d["crisp_on_top"],
                "trace_source_field": d.get("trace_source_field"),
                "error": d.get("error"),
            }
            for field in LABEL_FIELDS:
                row[field] = j.get(field) if j else None
            rows.append(row)
    return pl.DataFrame(rows)


def join_with_results(
    judgments_df: pl.DataFrame, results_path: Path
) -> pl.DataFrame:
    """Join judgments with original results to recover trial correctness."""
    results = load_results(results_path)
    join_keys = ["model", "prompt_id", "blur_px", "crisp_on_top"]
    # Keep only the correctness flag from results; aggregate to handle reps
    results_agg = (
        results.group_by(join_keys)
        .agg(pl.col("correct").mean().alias("correct_rate"))
    )
    return judgments_df.join(results_agg, on=join_keys, how="left")


def judgment_summary(judgments_path: Path) -> str:
    df = load_judgments(judgments_path)
    n_total = len(df)
    n_errors = int(df["error"].is_not_null().sum())
    n_valid = n_total - n_errors

    lines = [
        "## Trace judgment summary",
        "",
        f"- {n_total} traces judged ({n_errors} errors, {n_valid} valid)",
        "",
    ]

    for prompt in ("cot", "thinking"):
        sub = df.filter(
            (pl.col("prompt_id") == prompt) & pl.col("error").is_null()
        )
        if len(sub) == 0:
            continue

        lines.append(f"### {prompt.upper()} prompt --- bias-incongruent traces")
        lines.append(
            "Percentage of traces (per model) where the judge marked the label TRUE:"
        )
        lines.append("")

        header = (
            f"{'Model':22s} {'n':>5s} "
            + " ".join(f"{f.replace('_', ' ')[:14]:>15s}" for f in LABEL_FIELDS)
        )
        lines.append(header)
        lines.append("-" * len(header))

        models = sorted(sub["model"].unique().to_list())
        for model in models:
            ms = sub.filter(pl.col("model") == model)
            cells = [f"{model:22s} {len(ms):>5d}"]
            for field in LABEL_FIELDS:
                vals = ms[field].drop_nulls()
                pct = 100 * vals.sum() / len(vals) if len(vals) else 0.0
                cells.append(f"{pct:>14.1f}%")
            lines.append(" ".join(cells))

        # pooled row
        cells = [f"{'(pooled)':22s} {len(sub):>5d}"]
        for field in LABEL_FIELDS:
            vals = sub[field].drop_nulls()
            pct = 100 * vals.sum() / len(vals) if len(vals) else 0.0
            cells.append(f"{pct:>14.1f}%")
        lines.append(" ".join(cells))
        lines.append("")

    # cross-cutting comparison: thinking traces only,
    # within bias-incongruent --- did the model get the trial right?
    lines.append("### THINKING prompt --- correct vs incorrect traces")
    lines.append(
        "For thinking-prompt bias-incongruent trials, comparing label "
        "frequencies between traces that ended in the correct answer "
        "(blurred=in front) vs the incorrect (crisp=in front) answer."
    )
    lines.append(
        "Note: trial correctness is inferred from the judge's "
        "`identifies_correct_occluder` label; for a more authoritative "
        "split, join with the original results.jsonl on (model, prompt, "
        "blur, crisp_side, colour) keys."
    )
    lines.append("")

    return "\n".join(lines)
