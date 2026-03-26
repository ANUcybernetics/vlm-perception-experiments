"""Capture thinking traces for qualitative analysis.

Runs 4 representative conditions (crisp/blurred on top × left/right) with
Anthropic extended thinking and saves the full traces to a JSON file.
"""

import json
from pathlib import Path

import anthropic

from vlm_perception.evaluate import _encode_image, get_prompt
from vlm_perception.models import Colour, Condition, Side

STIMULI_DIR = Path("stimuli")
OUTPUT_PATH = Path("results/thinking_traces.json")

CONDITIONS = [
    Condition(crisp_on_top=True, crisp_side=Side.left, colour_crisp=Colour.red, colour_blurred=Colour.blue),
    Condition(crisp_on_top=True, crisp_side=Side.right, colour_crisp=Colour.green, colour_blurred=Colour.yellow),
    Condition(crisp_on_top=False, crisp_side=Side.left, colour_crisp=Colour.cyan, colour_blurred=Colour.magenta),
    Condition(crisp_on_top=False, crisp_side=Side.right, colour_crisp=Colour.blue, colour_blurred=Colour.red),
]

MODELS = [
    ("claude-sonnet-4-6", {"thinking": {"type": "enabled", "budget_tokens": 4096}, "max_tokens": 8192}),
    ("claude-opus-4-6", {"thinking": {"type": "adaptive"}, "max_tokens": 16000}),
]


def main():
    client = anthropic.Anthropic()
    prompt = get_prompt("thinking")
    traces = []

    for model, kwargs in MODELS:
        for cond in CONDITIONS:
            image_path = STIMULI_DIR / cond.image_filename
            b64 = _encode_image(image_path)
            print(f"{model} | {cond.image_filename} (correct: {cond.correct_answer.value})")

            response = client.messages.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                **kwargs,
            )

            thinking_text = ""
            answer_text = ""
            for block in response.content:
                if isinstance(block, anthropic.types.ThinkingBlock):
                    thinking_text = block.thinking
                elif isinstance(block, anthropic.types.TextBlock):
                    answer_text = block.text

            traces.append({
                "model": model,
                "condition": {
                    "crisp_on_top": cond.crisp_on_top,
                    "crisp_side": cond.crisp_side.value,
                    "colour_crisp": cond.colour_crisp.value,
                    "colour_blurred": cond.colour_blurred.value,
                    "correct_answer": cond.correct_answer.value,
                },
                "thinking": thinking_text,
                "answer": answer_text,
            })
            print(f"  answer: {answer_text.strip()}")
            print(f"  thinking: {thinking_text[:200]}...")
            print()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(traces, indent=2))
    print(f"Saved {len(traces)} traces to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
