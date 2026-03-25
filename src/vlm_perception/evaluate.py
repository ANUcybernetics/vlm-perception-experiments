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


def evaluate_anthropic(
    image_path: Path,
    condition: Condition,
    model: str = "claude-sonnet-4-6",
    prompt_id: str = DEFAULT_PROMPT_ID,
) -> TrialResult:
    prompt = get_prompt(prompt_id)
    client = anthropic.Anthropic()
    b64 = _encode_image(image_path)
    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=256,
                messages=[
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
            )
            break
        except anthropic.InternalServerError:
            if attempt == MAX_RETRIES - 1:
                raise
            delay = RETRY_BASE_DELAY * (2 ** attempt)
            log.warning("Anthropic 500 error, retrying in %.1fs (attempt %d/%d)", delay, attempt + 1, MAX_RETRIES)
            time.sleep(delay)
    block = response.content[0]
    raw: str = block.text if isinstance(block, anthropic.types.TextBlock) else ""
    parsed = _parse_response(raw)
    correct = parsed == condition.correct_answer if parsed else None
    return TrialResult(
        condition=condition,
        model=model,
        prompt_id=prompt_id,
        prompt=prompt,
        raw_response=raw,
        parsed_answer=parsed,
        correct=correct,
        timestamp=TrialResult.now(),
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
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                max_completion_tokens=256,
                messages=[
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
            )
            break
        except (openai.InternalServerError, openai.APIStatusError) as exc:
            if isinstance(exc, openai.APIStatusError) and exc.status_code < 500:
                raise
            if attempt == MAX_RETRIES - 1:
                raise
            delay = RETRY_BASE_DELAY * (2 ** attempt)
            log.warning("OpenAI server error, retrying in %.1fs (attempt %d/%d)", delay, attempt + 1, MAX_RETRIES)
            time.sleep(delay)
    raw = response.choices[0].message.content or ""
    parsed = _parse_response(raw)
    correct = parsed == condition.correct_answer if parsed else None
    return TrialResult(
        condition=condition,
        model=model,
        prompt_id=prompt_id,
        prompt=prompt,
        raw_response=raw,
        parsed_answer=parsed,
        correct=correct,
        timestamp=TrialResult.now(),
    )


def evaluate(
    image_path: Path,
    condition: Condition,
    provider: str,
    model: str,
    prompt_id: str = DEFAULT_PROMPT_ID,
) -> TrialResult:
    if provider == "anthropic":
        return evaluate_anthropic(image_path, condition, model=model, prompt_id=prompt_id)
    elif provider == "openai":
        return evaluate_openai(image_path, condition, model=model, prompt_id=prompt_id)
    else:
        raise ValueError(f"Unknown provider: {provider}")
