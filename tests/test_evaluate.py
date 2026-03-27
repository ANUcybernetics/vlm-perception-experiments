import asyncio

import pytest

from vlm_perception.evaluate import (
    MAX_RETRIES,
    RETRY_BASE_DELAY,
    _build_anthropic_request,
    _build_openai_request,
    _build_openai_responses_request,
    _extract_openai_responses_output,
    _make_trial_result,
    _parse_response,
    async_evaluate,
    get_prompt,
    load_prompts,
)
from vlm_perception.models import Colour, Condition, Side


def test_parse_json_left():
    assert _parse_response('{"answer": "left"}') == Side.left


def test_parse_json_right():
    assert _parse_response('Sure! {"answer": "right"}') == Side.right


def test_parse_freetext_left():
    assert _parse_response("The left circle is in front.") == Side.left


def test_parse_freetext_right():
    assert _parse_response("The right circle occludes the other.") == Side.right


def test_parse_ambiguous_returns_none():
    assert _parse_response("The left and right circles overlap.") is None


def test_parse_gibberish_returns_none():
    assert _parse_response("I cannot determine the answer.") is None


def test_load_prompts_has_expected_keys():
    prompts = load_prompts()
    assert "neutral" in prompts
    assert "minimal" in prompts
    assert "foreground" in prompts
    assert "psychophysics" in prompts


def test_get_prompt_returns_string():
    prompt = get_prompt("neutral")
    assert isinstance(prompt, str)
    assert len(prompt) > 0


def test_get_prompt_unknown_raises():
    with pytest.raises(ValueError, match="Unknown prompt"):
        get_prompt("nonexistent")


def test_retry_constants():
    assert MAX_RETRIES >= 1
    assert RETRY_BASE_DELAY > 0


def _make_condition():
    return Condition(
        crisp_on_top=True,
        crisp_side=Side.left,
        colour_crisp=Colour.red,
        colour_blurred=Colour.blue,
    )


def test_make_trial_result_correct():
    result = _make_trial_result(
        '{"answer": "left"}',
        _make_condition(),
        "test-model",
        "neutral",
        "test prompt",
    )
    assert result.parsed_answer == Side.left
    assert result.correct is True
    assert result.model == "test-model"
    assert result.reasoning_trace is None


def test_make_trial_result_incorrect():
    result = _make_trial_result(
        '{"answer": "right"}',
        _make_condition(),
        "test-model",
        "neutral",
        "test prompt",
    )
    assert result.parsed_answer == Side.right
    assert result.correct is False


def test_make_trial_result_unparseable():
    result = _make_trial_result(
        "gibberish",
        _make_condition(),
        "test-model",
        "neutral",
        "test prompt",
    )
    assert result.parsed_answer is None
    assert result.correct is None


def test_make_trial_result_with_reasoning_trace():
    result = _make_trial_result(
        '{"answer": "left"}',
        _make_condition(),
        "test-model",
        "thinking",
        "test prompt",
        reasoning_trace="I see the left circle has crisp edges...",
    )
    assert result.reasoning_trace == "I see the left circle has crisp edges..."
    assert result.correct is True


def test_build_anthropic_request_structure():
    req = _build_anthropic_request(
        "b64data", "prompt text", "neutral", "claude-sonnet-4-6"
    )
    assert req["model"] == "claude-sonnet-4-6"
    assert req["max_tokens"] == 1024
    assert len(req["messages"]) == 1
    assert len(req["messages"][0]["content"]) == 2


def test_build_anthropic_request_thinking():
    req = _build_anthropic_request(
        "b64data", "prompt text", "thinking", "claude-sonnet-4-6"
    )
    assert "thinking" in req
    assert req["thinking"]["type"] == "enabled"


def test_build_openai_request_structure():
    req = _build_openai_request("b64data", "prompt text", "neutral", "gpt-5.4")
    assert req["model"] == "gpt-5.4"
    assert req["max_completion_tokens"] == 4096
    assert len(req["messages"]) == 1


def test_build_openai_request_thinking():
    req = _build_openai_request("b64data", "prompt text", "thinking", "gpt-5.4")
    assert req["reasoning_effort"] == "medium"
    assert req["max_completion_tokens"] == 16384


def test_build_anthropic_request_thinking_opus_uses_enabled():
    req = _build_anthropic_request(
        "b64data", "prompt text", "thinking", "claude-opus-4-6"
    )
    assert req["thinking"]["type"] == "enabled"
    assert req["thinking"]["budget_tokens"] == 10000


def test_build_openai_responses_request_structure():
    req = _build_openai_responses_request("b64data", "prompt text", "gpt-5.4")
    assert req["model"] == "gpt-5.4"
    assert req["reasoning"]["effort"] == "medium"
    assert req["reasoning"]["summary"] == "auto"
    assert len(req["input"]) == 1
    content = req["input"][0]["content"]
    assert content[0]["type"] == "input_image"
    assert content[1]["type"] == "input_text"


def test_extract_openai_responses_output():
    from unittest.mock import MagicMock

    from openai.types.responses import ResponseOutputMessage, ResponseReasoningItem
    from openai.types.responses.response_output_message import ResponseOutputText
    from openai.types.responses.response_reasoning_item import Summary

    reasoning_item = ResponseReasoningItem(
        id="r1",
        type="reasoning",
        summary=[
            Summary(text="The left circle has crisp edges.", type="summary_text"),
            Summary(text="It appears to occlude the right.", type="summary_text"),
        ],
    )
    text_block = ResponseOutputText(
        type="output_text", text='{"answer": "left"}', annotations=[]
    )
    message_item = ResponseOutputMessage(
        id="m1",
        type="message",
        role="assistant",
        content=[text_block],
        status="completed",
    )
    response = MagicMock()
    response.output = [reasoning_item, message_item]

    raw, reasoning = _extract_openai_responses_output(response)
    assert raw == '{"answer": "left"}'
    assert "crisp edges" in reasoning
    assert "occlude" in reasoning


def test_extract_openai_responses_output_no_reasoning():
    from unittest.mock import MagicMock

    from openai.types.responses import ResponseOutputMessage
    from openai.types.responses.response_output_message import ResponseOutputText

    text_block = ResponseOutputText(
        type="output_text", text='{"answer": "right"}', annotations=[]
    )
    message_item = ResponseOutputMessage(
        id="m1",
        type="message",
        role="assistant",
        content=[text_block],
        status="completed",
    )
    response = MagicMock()
    response.output = [message_item]

    raw, reasoning = _extract_openai_responses_output(response)
    assert raw == '{"answer": "right"}'
    assert reasoning is None


def test_async_evaluate_unknown_provider():
    condition = _make_condition()

    async def _run():
        sem = asyncio.Semaphore(1)
        with pytest.raises(ValueError, match="Unknown provider"):
            await async_evaluate(
                Path("/fake"), condition, "unknown", "model", "neutral", sem
            )

    from pathlib import Path

    asyncio.run(_run())
