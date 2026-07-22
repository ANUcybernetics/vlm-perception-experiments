import pytest

from vlm_perception.models import Colour
from vlm_perception.scotoma import ScotomaStyle, pair_streams, render_scotoma


def test_pair_streams_basic():
    rows = pair_streams("AB", "XY")
    assert rows == [[("A", "X"), ("B", "Y")]]


def test_pair_streams_uppercases():
    rows = pair_streams("ab", "xy")
    assert rows == [[("A", "X"), ("B", "Y")]]


def test_real_space_is_blank_cell_consuming_nothing():
    rows = pair_streams("A B", "XY")
    assert rows == [[("A", "X"), (None, None), ("B", "Y")]]


def test_robot_space_leaves_real_glyph_unpartnered():
    rows = pair_streams("ABC", "X Z")
    assert rows == [[("A", "X"), ("B", None), ("C", "Z")]]


def test_real_newlines_break_rows():
    rows = pair_streams("AB\nCD", "WXYZ")
    assert rows == [
        [("A", "W"), ("B", "X")],
        [("C", "Y"), ("D", "Z")],
    ]


def test_robot_newlines_stripped():
    rows = pair_streams("ABCD", "WX\nYZ")
    assert rows == [[("A", "W"), ("B", "X"), ("C", "Y"), ("D", "Z")]]


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        pair_streams("ABC", "XY")


def test_empty_real_raises():
    with pytest.raises(ValueError, match="empty"):
        render_scotoma("  ", "")


def test_render_returns_rgb_image():
    img = render_scotoma("HI", "NO")
    assert img.mode == "RGB"
    assert img.width > 0 and img.height > 0


def test_render_deterministic():
    a = render_scotoma("HELLO", "WORLD")
    b = render_scotoma("HELLO", "WORLD")
    assert a.tobytes() == b.tobytes()


def test_robot_stream_changes_pixels():
    a = render_scotoma("HELLO", "WORLD")
    b = render_scotoma("HELLO", "MUNGO")
    assert a.tobytes() != b.tobytes()


def test_zero_blur_differs_from_default():
    crisp = render_scotoma("HI", "NO", ScotomaStyle(blur_fraction=0.0))
    blurred = render_scotoma("HI", "NO")
    assert crisp.tobytes() != blurred.tobytes()


def test_depth_flip_changes_pixels():
    front = render_scotoma("HI", "NO", ScotomaStyle(blurred_on_top=True))
    back = render_scotoma("HI", "NO", ScotomaStyle(blurred_on_top=False))
    assert front.tobytes() != back.tobytes()


def test_multiline_taller_than_single_line():
    one = render_scotoma("ABCD", "WXYZ")
    two = render_scotoma("AB\nCD", "WXYZ")
    assert two.height > one.height
    assert two.width < one.width


def test_style_colour_roundtrip():
    style = ScotomaStyle(colour_real=Colour.magenta, colour_robot=Colour.green)
    img = render_scotoma("A", "B", style)
    assert img.mode == "RGB"
