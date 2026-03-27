import asyncio
import base64
import json
import logging
import re
import time
from pathlib import Path

import anthropic
import openai

from vlm_perception.models import Condition, Side, TrialResult

log = logging.getLogger(__name__)

MAX_RETRIES = 5
RETRY_BASE_DELAY = 2.0

PROMPTS_PATH = Path(__file__).parent / "prompts.json"
DEFAULT_PROMPT_ID = "neutral"


def load_prompts() -> dict[str, str]:
    return json.loads(PROMPTS_PATH.read_text())


def get_prompt(prompt_id: str) -> str:
    prompts = load_prompts()
    if prompt_id not in prompts:
        available = ", ".join(prompts)
        raise ValueError(f"Unknown prompt: {prompt_id!r}. Available: {available}")
    return prompts[prompt_id]


def _encode_image(path: Path) -> str:
    return base64.standard_b64encode(path.read_bytes()).decode()


def _parse_response(text: str) -> Side | None:
    text_lower = text.lower()
    json_match = re.search(r'"answer"\s*:\s*"(left|right)"', text_lower)
    if json_match:
        return Side(json_match.group(1))
    if "left" in text_lower and "right" not in text_lower:
        return Side.left
    if "right" in text_lower and "left" not in text_lower:
        return Side.right
    return None


THINKING_PROMPT_ID = "thinking"


def _anthropic_thinking_kwargs(model: str) -> dict:
    if model == "claude-opus-4-6":
        return {"thinking": {"type": "enabled", "budget_tokens": 10000}, "max_tokens": 16000}
    return {"thinking": {"type": "enabled", "budget_tokens": 4096}, "max_tokens": 8192}


def _build_anthropic_request(b64: str, prompt: str, prompt_id: str, model: str) -> dict:
    api_kwargs: dict = {}
    if prompt_id == THINKING_PROMPT_ID:
        api_kwargs = _anthropic_thinking_kwargs(model)
    else:
        api_kwargs = {"max_tokens": 1024}
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        **api_kwargs,
    }


def _build_openai_request(b64: str, prompt: str, prompt_id: str, model: str) -> dict:
    extra: dict = {}
    if prompt_id == THINKING_PROMPT_ID:
        extra["reasoning_effort"] = "medium"
        extra["max_completion_tokens"] = 16384
    else:
        extra["max_completion_tokens"] = 4096
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        **extra,
    }


def _build_openai_responses_request(b64: str, prompt: str, model: str) -> dict:
    return {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{b64}",
                    },
                    {"type": "input_text", "text": prompt},
                ],
            }
        ],
        "reasoning": {"effort": "medium", "summary": "auto"},
        "max_output_tokens": 16384,
    }


def _extract_openai_responses_output(
    response: "openai.types.responses.Response",
) -> tuple[str, str | None]:
    from openai.types.responses import ResponseOutputMessage, ResponseReasoningItem

    raw = ""
    reasoning_parts: list[str] = []
    for item in response.output:
        if isinstance(item, ResponseReasoningItem):
            for summary in item.summary:
                reasoning_parts.append(summary.text)
        elif isinstance(item, ResponseOutputMessage):
            for block in item.content:
                if block.type == "output_text":
                    raw = block.text
    reasoning = "\n".join(reasoning_parts) if reasoning_parts else None
    return raw, reasoning


def _extract_anthropic_response(
    response: anthropic.types.Message,
) -> tuple[str, str | None]:
    raw = ""
    thinking = ""
    for block in response.content:
        if isinstance(block, anthropic.types.ThinkingBlock):
            thinking += block.thinking
        elif isinstance(block, anthropic.types.TextBlock):
            raw = block.text
    return raw, thinking or None


def _make_trial_result(
    raw: str,
    condition: Condition,
    model: str,
    prompt_id: str,
    prompt: str,
    reasoning_trace: str | None = None,
) -> TrialResult:
    parsed = _parse_response(raw)
    correct = parsed == condition.correct_answer if parsed else None
    return TrialResult(
        condition=condition,
        model=model,
        prompt_id=prompt_id,
        prompt=prompt,
        raw_response=raw,
        reasoning_trace=reasoning_trace,
        parsed_answer=parsed,
        correct=correct,
        timestamp=TrialResult.now(),
    )


def evaluate_anthropic(
    image_path: Path,
    condition: Condition,
    model: str = "claude-sonnet-4-6",
    prompt_id: str = DEFAULT_PROMPT_ID,
) -> TrialResult:
    prompt = get_prompt(prompt_id)
    client = anthropic.Anthropic()
    b64 = _encode_image(image_path)
    request = _build_anthropic_request(b64, prompt, prompt_id, model)
    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(**request)
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
            time.sleep(delay)
    raw, thinking = _extract_anthropic_response(response)
    return _make_trial_result(
        raw, condition, model, prompt_id, prompt,
        reasoning_trace=thinking,
    )


def evaluate_openai(
    image_path: Path,
    condition: Condition,
    model: str = "gpt-5.4",
    prompt_id: str = DEFAULT_PROMPT_ID,
) -> TrialResult:
    prompt = get_prompt(prompt_id)
    client = openai.OpenAI()
    b64 = _encode_image(image_path)
    use_responses = prompt_id == THINKING_PROMPT_ID
    if use_responses:
        request = _build_openai_responses_request(b64, prompt, model)
    else:
        request = _build_openai_request(b64, prompt, prompt_id, model)
    for attempt in range(MAX_RETRIES):
        try:
            if use_responses:
                response = client.responses.create(**request)
            else:
                response = client.chat.completions.create(**request)
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
            time.sleep(delay)
    if use_responses:
        raw, reasoning = _extract_openai_responses_output(response)
    else:
        raw = response.choices[0].message.content or ""
        reasoning = None
    return _make_trial_result(
        raw, condition, model, prompt_id, prompt, reasoning_trace=reasoning
    )


async def async_evaluate_anthropic(
    image_path: Path,
    condition: Condition,
    model: str,
    prompt_id: str,
    semaphore: asyncio.Semaphore,
) -> TrialResult:
    prompt = get_prompt(prompt_id)
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
    raw, thinking = _extract_anthropic_response(response)
    return _make_trial_result(
        raw, condition, model, prompt_id, prompt,
        reasoning_trace=thinking,
    )


async def async_evaluate_openai(
    image_path: Path,
    condition: Condition,
    model: str,
    prompt_id: str,
    semaphore: asyncio.Semaphore,
) -> TrialResult:
    prompt = get_prompt(prompt_id)
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
        raw, reasoning = _extract_openai_responses_output(response)
    else:
        raw = response.choices[0].message.content or ""
        reasoning = None
    return _make_trial_result(
        raw, condition, model, prompt_id, prompt, reasoning_trace=reasoning
    )


async def async_evaluate(
    image_path: Path,
    condition: Condition,
    provider: str,
    model: str,
    prompt_id: str,
    semaphore: asyncio.Semaphore,
) -> TrialResult:
    if provider == "anthropic":
        return await async_evaluate_anthropic(
            image_path, condition, model, prompt_id, semaphore
        )
    elif provider == "openai":
        return await async_evaluate_openai(
            image_path, condition, model, prompt_id, semaphore
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")


def evaluate(
    image_path: Path,
    condition: Condition,
    provider: str,
    model: str,
    prompt_id: str = DEFAULT_PROMPT_ID,
) -> TrialResult:
    if provider == "anthropic":
        return evaluate_anthropic(
            image_path, condition, model=model, prompt_id=prompt_id
        )
    elif provider == "openai":
        return evaluate_openai(image_path, condition, model=model, prompt_id=prompt_id)
    else:
        raise ValueError(f"Unknown provider: {provider}")
