"""Pillow renderer for the Scotoma typeface."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from vlm_perception.scotoma.models import (
    Layout,
    ScotomaStyle,
    pair_aligned,
    pair_streams,
)

FONT_PATH = Path(__file__).parent / "fonts" / "Jost-wght.ttf"


def _load_font(style: ScotomaStyle) -> ImageFont.FreeTypeFont:
    font = ImageFont.truetype(FONT_PATH, style.font_size)
    font.set_variation_by_axes([style.weight])
    return font


def _cell_metrics(
    rows: Layout,
    font: ImageFont.FreeTypeFont,
    style: ScotomaStyle,
) -> tuple[float, float]:
    """Fixed cell size: widest glyph in either stream, plus tracking."""
    glyphs = {g for row in rows for cell in row for g in cell if g is not None}
    max_advance = max((font.getlength(g) for g in glyphs), default=style.font_size)
    cell_w = max_advance + style.tracking_fraction * style.font_size
    cell_h = style.line_height_fraction * style.font_size
    return cell_w, cell_h


def render_scotoma(
    real: str, robot: str, style: ScotomaStyle | None = None
) -> Image.Image:
    """Render the two streams as a single Scotoma image.

    Each layer's glyphs are offset symmetrically left and right about the
    cell centre (real to the right, robot to the left) but share a common
    baseline, so neither stream sits "on grid" and vertical position gives
    no depth cue --- only the blur does. This mirrors the circle stimuli,
    which overlapped horizontally at a common vertical centre.
    """
    if style is None:
        style = ScotomaStyle()
    if not real.strip():
        raise ValueError("Real text is empty")
    return _render_rows(pair_streams(real, robot), style)


def render_solo(text: str, style: ScotomaStyle | None = None) -> Image.Image:
    """Render the real stream alone (legibility baseline, no robot layer)."""
    if style is None:
        style = ScotomaStyle()
    if not text.strip():
        raise ValueError("Text is empty")
    rows: Layout = [
        [(c if c != " " else None, None) for c in line]
        for line in text.upper().split("\n")
    ]
    return _render_rows(rows, style)


def _render_rows(rows: Layout, style: ScotomaStyle) -> Image.Image:
    """Composite already-paired rows of cells into a Scotoma image."""
    font = _load_font(style)
    cell_w, cell_h = _cell_metrics(rows, font, style)

    pad = style.font_size * 0.75
    width = round(2 * pad + cell_w * max(len(row) for row in rows))
    height = round(2 * pad + cell_h * len(rows))

    real_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    robot_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw_real = ImageDraw.Draw(real_layer)
    draw_robot = ImageDraw.Draw(robot_layer)
    d = style.offset_px / 2

    for row_i, row in enumerate(rows):
        cy = pad + cell_h * (row_i + 0.5)
        for col_i, (real_char, robot_char) in enumerate(row):
            cx = pad + cell_w * (col_i + 0.5)
            if real_char is not None:
                draw_real.text(
                    (cx + d, cy),
                    real_char,
                    font=font,
                    fill=(*style.colour_real.rgb, 255),
                    anchor="mm",
                )
            if robot_char is not None:
                draw_robot.text(
                    (cx - d, cy),
                    robot_char,
                    font=font,
                    fill=(*style.colour_robot.rgb, 255),
                    anchor="mm",
                )

    if style.blur_px > 0:
        real_layer = real_layer.filter(ImageFilter.GaussianBlur(radius=style.blur_px))

    bg = style.background_grey
    canvas = Image.new("RGBA", (width, height), (bg, bg, bg, 255))
    back, front = (
        (robot_layer, real_layer) if style.blurred_on_top else (real_layer, robot_layer)
    )
    canvas = Image.alpha_composite(canvas, back)
    canvas = Image.alpha_composite(canvas, front)
    return canvas.convert("RGB")


def render_diptych(
    top: str, bottom: str, style: ScotomaStyle | None = None
) -> Image.Image:
    """Render the reciprocal diptych of two strings.

    The Scotoma encoding is symmetric, so the same pair yields two panels
    that swap who reads what. The top panel puts ``top`` in the blurred
    foreground (humans read ``top``, VLMs read ``bottom``); the bottom
    panel reverses the roles (humans read ``bottom``, VLMs read ``top``).
    Read down the human column and the machine column and the two
    messages are exchanged.
    """
    if style is None:
        style = ScotomaStyle()
    if not top.strip() or not bottom.strip():
        raise ValueError("Both diptych messages must be non-empty")
    panel_top = _render_rows(pair_aligned(top, bottom), style)
    panel_bottom = _render_rows(pair_aligned(bottom, top), style)

    gap = round(style.font_size * 0.5)
    bg = style.background_grey
    width = max(panel_top.width, panel_bottom.width)
    height = panel_top.height + gap + panel_bottom.height
    canvas = Image.new("RGB", (width, height), (bg, bg, bg))
    canvas.paste(panel_top, ((width - panel_top.width) // 2, 0))
    canvas.paste(
        panel_bottom, ((width - panel_bottom.width) // 2, panel_top.height + gap)
    )
    return canvas
