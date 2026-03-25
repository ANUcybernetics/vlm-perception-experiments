import pytest

from vlm_perception.evaluate import _parse_response, get_prompt, load_prompts
from vlm_perception.models import Side


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
