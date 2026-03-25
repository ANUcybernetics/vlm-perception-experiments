import pytest

from vlm_perception.models import (
    MODEL_REGISTRY,
    Colour,
    Condition,
    Side,
    all_conditions,
    resolve_model,
)


def test_colour_rgb_in_gamut():
    for colour in Colour:
        r, g, b = colour.rgb
        assert 0 <= r <= 255
        assert 0 <= g <= 255
        assert 0 <= b <= 255


def test_colours_are_distinct():
    rgbs = [c.rgb for c in Colour]
    assert len(set(rgbs)) == 6


def test_correct_answer_crisp_on_top():
    c = Condition(
        crisp_on_top=True,
        crisp_side=Side.left,
        colour_crisp=Colour.red,
        colour_blurred=Colour.blue,
    )
    assert c.correct_answer == Side.left


def test_correct_answer_blurred_on_top():
    c = Condition(
        crisp_on_top=False,
        crisp_side=Side.left,
        colour_crisp=Colour.red,
        colour_blurred=Colour.blue,
    )
    assert c.correct_answer == Side.right


def test_all_conditions_count():
    conditions = all_conditions()
    assert len(conditions) == 2 * 2 * 6 * 5  # 120


def test_no_same_colour_conditions():
    for c in all_conditions():
        assert c.colour_crisp != c.colour_blurred


def test_image_filename_format():
    c = Condition(
        crisp_on_top=True,
        crisp_side=Side.left,
        colour_crisp=Colour.red,
        colour_blurred=Colour.blue,
    )
    assert c.image_filename == "crisp-top_left_red_blue.png"


def test_resolve_model_valid():
    spec = resolve_model("claude-sonnet-4-6")
    assert spec.provider == "anthropic"
    assert spec.model_id == "claude-sonnet-4-6"


def test_resolve_model_openai():
    spec = resolve_model("gpt-5.4-mini")
    assert spec.provider == "openai"
    assert spec.model_id == "gpt-5.4-mini"


def test_resolve_model_unknown():
    with pytest.raises(ValueError, match="Unknown model"):
        resolve_model("nonexistent-model")


def test_model_registry_all_have_provider():
    for name, spec in MODEL_REGISTRY.items():
        assert spec.provider in ("anthropic", "openai"), f"{name} has invalid provider"
