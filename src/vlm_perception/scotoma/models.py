"""Data models and stream-pairing logic for the Scotoma typeface.

Scotoma renders two text streams into one image: a "real" stream (for
humans) and a "robot" stream (for VLMs). Each character cell overlays one
glyph from each stream; one layer is Gaussian-blurred and composited on
top of the other. Occlusion edge blur makes humans read the blurred
foreground glyph as the intact letterform, while VLMs (per the MAD'26
findings) apply a "sharp means closer" heuristic and read the crisp one.
"""

from pydantic import BaseModel, Field

from vlm_perception.models import Colour

# A cell pairs an optional real glyph with an optional robot glyph.
# (None, None) is a blank cell from a space in the real stream.
Cell = tuple[str | None, str | None]
# A row is a horizontal run of cells; a layout is a list of rows.
Row = list[Cell]
Layout = list[Row]


class ScotomaStyle(BaseModel):
    """Rendering parameters, expressed relative to font size where possible."""

    font_size: int = Field(default=96, gt=0)
    weight: int = Field(default=900, ge=100, le=900)
    blur_fraction: float = Field(default=0.07, ge=0.0)
    offset_fraction: float = Field(default=0.38, ge=0.0)
    tracking_fraction: float = Field(default=0.12, ge=0.0)
    line_height_fraction: float = Field(default=1.3, gt=0.0)
    colour_real: Colour = Colour.red
    colour_robot: Colour = Colour.cyan
    background_grey: int = Field(default=128, ge=0, le=255)
    # True = real (blurred) layer composited in front: the exploit.
    # False = crisp robot layer in front: the congruent control.
    blurred_on_top: bool = True

    @property
    def blur_px(self) -> float:
        return self.blur_fraction * self.font_size

    @property
    def offset_px(self) -> float:
        return self.offset_fraction * self.font_size


def pair_streams(real: str, robot: str) -> Layout:
    """Pair the two streams into rows of glyph cells.

    The real stream drives layout: newlines break rows, spaces emit blank
    cells (consuming no robot character), and every other character emits
    a cell that consumes exactly one robot character. Newlines in the
    robot stream are stripped; a robot space yields an unpartnered real
    glyph, preserving the robot stream's word boundaries.
    """
    real = real.upper()
    robot = robot.upper().replace("\n", "")

    n_printable = sum(1 for c in real if c not in (" ", "\n"))
    if len(robot) != n_printable:
        raise ValueError(
            f"Stream length mismatch: real has {n_printable} printable "
            f"characters but robot has {len(robot)} (after stripping newlines)"
        )

    rows: Layout = []
    robot_chars = iter(robot)
    for line in real.split("\n"):
        row: Row = []
        for char in line:
            if char == " ":
                row.append((None, None))
            else:
                partner = next(robot_chars)
                row.append((char, partner if partner != " " else None))
        rows.append(row)
    return rows


def pair_aligned(real: str, robot: str) -> Layout:
    """Pair two equal-length streams position-by-position (symmetric).

    Unlike ``pair_streams``, neither stream drives layout: cell *i* holds
    ``real[i]`` and ``robot[i]``, with spaces rendered as gaps in their own
    stream. The two strings must be the same length and break lines at the
    same positions. This is the pairing used by the reciprocal diptych,
    where the same pair must encode cleanly in both role orderings.
    """
    real = real.upper()
    robot = robot.upper()
    if len(real) != len(robot):
        raise ValueError(
            f"Aligned streams must be equal length: {len(real)} vs {len(robot)}"
        )

    rows: Layout = [[]]
    for r_char, b_char in zip(real, robot, strict=True):
        if r_char == "\n" or b_char == "\n":
            if r_char != b_char:
                raise ValueError(
                    "Aligned streams must break lines at the same position"
                )
            rows.append([])
            continue
        rows[-1].append(
            (
                r_char if r_char != " " else None,
                b_char if b_char != " " else None,
            )
        )
    return rows
