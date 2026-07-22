"""Async VLM transcription evaluation for the Scotoma experiment.

Reuses the provider request builders and response extractors from the
circle experiment (vlm_perception.evaluate); only the prompt registry and
the scoring differ. The "thinking" prompt id triggers provider-level
reasoning exactly as in the circle evaluate path.
"""

import asyncio
import json
import logging
from pathlib import Path

import anthropic
import openai

from vlm_perception.evaluate import (
    MAX_RETRIES,
    RETRY_BASE_DELAY,
    THINKING_PROMPT_ID,
    _build_anthropic_request,
    _build_openai_request,
    _build_openai_responses_request,
    _encode_image,
    _extract_anthropic_response,
    _extract_openai_responses_output,
)
from vlm_perception.scotoma.experiment import ScotomaCondition, ScotomaTrialResult
from vlm_perception.scotoma.scoring import score_transcription

log = logging.getLogger(__name__)

PROMPTS_PATH = Path(__file__).parent / "prompts.json"
DEFAULT_PROMPT_ID = "naive"


def load_prompts() -> dict[str, str]:
    return json.loads(PROMPTS_PATH.read_text())


def get_prompt(prompt_id: str) -> str:
    prompts = load_prompts()
    if prompt_id not in prompts:
        available = ", ".join(prompts)
        raise ValueError(f"Unknown prompt: {prompt_id!r}. Available: {available}")
    return prompts[prompt_id]


def _make_trial_result(
    raw: str,
    condition: ScotomaCondition,
    model: str,
    prompt_id: str,
    prompt: str,
    reasoning_trace: str | None = None,
) -> ScotomaTrialResult:
    score = score_transcription(raw, condition.string_real, condition.string_robot)
    return ScotomaTrialResult(
        condition=condition,
        model=model,
        prompt_id=prompt_id,
        prompt=prompt,
        raw_response=raw,
        reasoning_trace=reasoning_trace,
        raw_transcription=score.raw_transcription,
        dist_real_lev=score.dist_real_lev,
        dist_robot_lev=score.dist_robot_lev,
        dist_real_ham=score.dist_real_ham,
        dist_robot_ham=score.dist_robot_ham,
        bias_index_lev=score.bias_index_lev,
        bias_index_ham=score.bias_index_ham,
        timestamp=ScotomaTrialResult.now(),
    )


async def _anthropic_transcribe(
    image_path: Path,
    model: str,
    prompt: str,
    prompt_id: str,
    semaphore: asyncio.Semaphore,
) -> tuple[str, str | None]:
    b64 = _encode_image(image_path)
    request = _build_anthropic_request(b64, prompt, prompt_id, model)
    client = anthropic.AsyncAnthropic()
    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                response = await client.messages.create(**request)
                break
            except anthropic.InternalServerError:
                if attempt == MAX_RETRIES - 1:
                    raise
                delay = RETRY_BASE_DELAY * (2**attempt)
                log.warning(
                    "Anthropic 500 error, retrying in %.1fs (attempt %d/%d)",
                    delay,
                    attempt + 1,
                    MAX_RETRIES,
                )
                await asyncio.sleep(delay)
    return _extract_anthropic_response(response)


async def _openai_transcribe(
    image_path: Path,
    model: str,
    prompt: str,
    prompt_id: str,
    semaphore: asyncio.Semaphore,
) -> tuple[str, str | None]:
    b64 = _encode_image(image_path)
    use_responses = prompt_id == THINKING_PROMPT_ID
    if use_responses:
        request = _build_openai_responses_request(b64, prompt, model)
    else:
        request = _build_openai_request(b64, prompt, prompt_id, model)
    client = openai.AsyncOpenAI()
    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                if use_responses:
                    response = await client.responses.create(**request)
                else:
                    response = await client.chat.completions.create(**request)
                break
            except (openai.InternalServerError, openai.APIStatusError) as exc:
                if isinstance(exc, openai.APIStatusError) and exc.status_code < 500:
                    raise
                if attempt == MAX_RETRIES - 1:
                    raise
                delay = RETRY_BASE_DELAY * (2**attempt)
                log.warning(
                    "OpenAI server error, retrying in %.1fs (attempt %d/%d)",
                    delay,
                    attempt + 1,
                    MAX_RETRIES,
                )
                await asyncio.sleep(delay)
    if use_responses:
        return _extract_openai_responses_output(response)
    return response.choices[0].message.content or "", None


async def async_evaluate_scotoma(
    image_path: Path,
    condition: ScotomaCondition,
    provider: str,
    model: str,
    prompt_id: str,
    semaphore: asyncio.Semaphore,
) -> ScotomaTrialResult:
    prompt = get_prompt(prompt_id)
    if provider == "anthropic":
        raw, reasoning = await _anthropic_transcribe(
            image_path, model, prompt, prompt_id, semaphore
        )
    elif provider == "openai":
        raw, reasoning = await _openai_transcribe(
            image_path, model, prompt, prompt_id, semaphore
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")
    return _make_trial_result(
        raw, condition, model, prompt_id, prompt, reasoning_trace=reasoning
    )
