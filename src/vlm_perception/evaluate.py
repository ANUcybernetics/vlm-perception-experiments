import base64
import re
from pathlib import Path

import anthropic
import openai

from vlm_perception.models import Condition, Side, TrialResult

PROMPT = (
    "This image shows two overlapping circles on a grey background. "
    "One circle is on the left and one is on the right. "
    "One circle is in front of (occluding) the other. "
    "Which circle is in front --- the left circle or the right circle? "
    'Reply with a single JSON object: {"answer": "left"} or {"answer": "right"}.'
)


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
    model: str = "claude-opus-4-6-20250415",
    prompt: str = PROMPT,
) -> TrialResult:
    client = anthropic.Anthropic()
    b64 = _encode_image(image_path)
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
    block = response.content[0]
    raw: str = block.text if isinstance(block, anthropic.types.TextBlock) else ""
    parsed = _parse_response(raw)
    correct = parsed == condition.correct_answer if parsed else None
    return TrialResult(
        condition=condition,
        model=model,
        prompt=prompt,
        raw_response=raw,
        parsed_answer=parsed,
        correct=correct,
        timestamp=TrialResult.now(),
    )


def evaluate_openai(
    image_path: Path,
    condition: Condition,
    model: str = "gpt-4o",
    prompt: str = PROMPT,
) -> TrialResult:
    client = openai.OpenAI()
    b64 = _encode_image(image_path)
    response = client.chat.completions.create(
        model=model,
        max_tokens=256,
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
    raw = response.choices[0].message.content or ""
    parsed = _parse_response(raw)
    correct = parsed == condition.correct_answer if parsed else None
    return TrialResult(
        condition=condition,
        model=model,
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
    prompt: str = PROMPT,
) -> TrialResult:
    if provider == "anthropic":
        return evaluate_anthropic(image_path, condition, model=model, prompt=prompt)
    elif provider == "openai":
        return evaluate_openai(image_path, condition, model=model, prompt=prompt)
    else:
        raise ValueError(f"Unknown provider: {provider}")
