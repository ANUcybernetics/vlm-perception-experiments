from vlm_perception.models import Colour, Condition, Side, all_conditions


def test_colour_rgb_values():
    assert Colour.red.rgb == (255, 0, 0)
    assert Colour.cyan.rgb == (0, 255, 255)


def test_correct_answer_crisp_on_top():
    c = Condition(crisp_on_top=True, crisp_side=Side.left, colour_crisp=Colour.red, colour_blurred=Colour.blue)
    assert c.correct_answer == Side.left


def test_correct_answer_blurred_on_top():
    c = Condition(crisp_on_top=False, crisp_side=Side.left, colour_crisp=Colour.red, colour_blurred=Colour.blue)
    assert c.correct_answer == Side.right


def test_all_conditions_count():
    conditions = all_conditions()
    assert len(conditions) == 2 * 2 * 6 * 5  # 120


def test_no_same_colour_conditions():
    for c in all_conditions():
        assert c.colour_crisp != c.colour_blurred


def test_image_filename_format():
    c = Condition(crisp_on_top=True, crisp_side=Side.left, colour_crisp=Colour.red, colour_blurred=Colour.blue)
    assert c.image_filename == "crisp-top_left_red_blue.png"
