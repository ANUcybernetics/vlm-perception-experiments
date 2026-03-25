from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from vlm_perception.models import Condition, Side, all_conditions

CANVAS_SIZE = 512
CIRCLE_RADIUS = 100
OVERLAP_OFFSET = 75
BLUR_RADIUS = 20
BG_GREY = (128, 128, 128)


def _circle_centre(side: Side) -> tuple[int, int]:
    cy = CANVAS_SIZE // 2
    cx_left = CANVAS_SIZE // 2 - OVERLAP_OFFSET
    cx_right = CANVAS_SIZE // 2 + OVERLAP_OFFSET
    if side == Side.left:
        return (cx_left, cy)
    return (cx_right, cy)


def _draw_circle(
    colour: tuple[int, int, int],
    centre: tuple[int, int],
    blur: bool,
) -> Image.Image:
    """Draw a single circle on a transparent layer, optionally blurred."""
    layer = Image.new("RGBA", (CANVAS_SIZE, CANVAS_SIZE), (*colour, 0))
    draw = ImageDraw.Draw(layer)
    x, y = centre
    bbox = (x - CIRCLE_RADIUS, y - CIRCLE_RADIUS, x + CIRCLE_RADIUS, y + CIRCLE_RADIUS)
    draw.ellipse(bbox, fill=(*colour, 255))
    if blur:
        layer = layer.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS))
    return layer


def generate_image(condition: Condition) -> Image.Image:
    """Generate a stimulus image for the given condition."""
    canvas = Image.new("RGBA", (CANVAS_SIZE, CANVAS_SIZE), (*BG_GREY, 255))

    crisp_centre = _circle_centre(condition.crisp_side)
    blurred_side = Side.right if condition.crisp_side == Side.left else Side.left
    blurred_centre = _circle_centre(blurred_side)

    crisp_layer = _draw_circle(condition.colour_crisp.rgb, crisp_centre, blur=False)
    blurred_layer = _draw_circle(
        condition.colour_blurred.rgb, blurred_centre, blur=True
    )

    if condition.crisp_on_top:
        canvas = Image.alpha_composite(canvas, blurred_layer)
        canvas = Image.alpha_composite(canvas, crisp_layer)
    else:
        canvas = Image.alpha_composite(canvas, crisp_layer)
        canvas = Image.alpha_composite(canvas, blurred_layer)

    return canvas.convert("RGB")


def generate_all(output_dir: Path) -> list[Path]:
    """Generate all factorial stimulus images, returning paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for condition in all_conditions():
        path = output_dir / condition.image_filename
        img = generate_image(condition)
        img.save(path)
        paths.append(path)
    return paths
