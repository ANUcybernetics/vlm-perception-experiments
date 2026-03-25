from vlm_perception.models import Colour, Condition, Side
from vlm_perception.stimuli import CANVAS_SIZE, generate_image


def test_generate_image_size():
    c = Condition(
        crisp_on_top=True,
        crisp_side=Side.left,
        colour_crisp=Colour.red,
        colour_blurred=Colour.blue,
    )
    img = generate_image(c)
    assert img.size == (CANVAS_SIZE, CANVAS_SIZE)
    assert img.mode == "RGB"


def test_generate_image_not_uniform():
    c = Condition(
        crisp_on_top=False,
        crisp_side=Side.right,
        colour_crisp=Colour.green,
        colour_blurred=Colour.magenta,
    )
    img = generate_image(c)
    pixels = set(img.get_flattened_data())
    assert len(pixels) > 1


def test_different_conditions_produce_different_images():
    c1 = Condition(
        crisp_on_top=True,
        crisp_side=Side.left,
        colour_crisp=Colour.red,
        colour_blurred=Colour.blue,
    )
    c2 = Condition(
        crisp_on_top=False,
        crisp_side=Side.left,
        colour_crisp=Colour.red,
        colour_blurred=Colour.blue,
    )
    img1 = generate_image(c1)
    img2 = generate_image(c2)
    assert list(img1.get_flattened_data()) != list(img2.get_flattened_data())
