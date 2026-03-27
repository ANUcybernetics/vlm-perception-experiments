import pytest

from vlm_perception.models import (
    BLUR_SWEEP_COLOUR_PAIRS,
    BLUR_SWEEP_RADII,
    MODEL_REGISTRY,
    Colour,
    Condition,
    Side,
    all_conditions,
    blur_sweep_conditions,
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
    assert c.image_filename == "crisp-top_left_red_blue_blur20.png"


def test_image_filename_custom_blur():
    c = Condition(
        crisp_on_top=True,
        crisp_side=Side.left,
        colour_crisp=Colour.red,
        colour_blurred=Colour.blue,
        blur_radius=8,
    )
    assert c.image_filename == "crisp-top_left_red_blue_blur8.png"


def test_blur_sweep_conditions_count():
    conditions = blur_sweep_conditions()
    expected = (
        len(BLUR_SWEEP_RADII)
        * 2  # depth order
        * 2  # side
        * len(BLUR_SWEEP_COLOUR_PAIRS)
    )
    assert len(conditions) == expected
    assert len(conditions) == 96


def test_blur_sweep_conditions_blur_values():
    conditions = blur_sweep_conditions()
    blur_values = sorted({c.blur_radius for c in conditions})
    assert blur_values == sorted(BLUR_SWEEP_RADII)


def test_blur_sweep_conditions_colour_pairs():
    conditions = blur_sweep_conditions()
    pairs = {(c.colour_crisp, c.colour_blurred) for c in conditions}
    assert pairs == set(BLUR_SWEEP_COLOUR_PAIRS)


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
