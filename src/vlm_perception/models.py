from enum import Enum
from datetime import datetime, timezone

from pydantic import BaseModel


OKLCH_LIGHTNESS = 0.7
OKLCH_CHROMA = 0.15


def _oklch_to_rgb(L: float, C: float, h_deg: float) -> tuple[int, int, int]:
    """Convert OKLCH to sRGB, clamping to [0, 255]."""
    import math

    h_rad = math.radians(h_deg)
    a = C * math.cos(h_rad)
    b = C * math.sin(h_rad)

    # OKLab to linear sRGB via the LMS intermediate
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b

    l = l_ * l_ * l_
    m = m_ * m_ * m_
    s = s_ * s_ * s_

    r_lin = +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g_lin = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    b_lin = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s

    def linear_to_srgb(x: float) -> int:
        if x <= 0.0031308:
            v = 12.92 * x
        else:
            v = 1.055 * (x ** (1.0 / 2.4)) - 0.055
        return max(0, min(255, round(v * 255)))

    return (linear_to_srgb(r_lin), linear_to_srgb(g_lin), linear_to_srgb(b_lin))


class Colour(str, Enum):
    red = "red"
    yellow = "yellow"
    green = "green"
    cyan = "cyan"
    blue = "blue"
    magenta = "magenta"

    @property
    def oklch_hue(self) -> float:
        """OKLCH hue angle (degrees), 6 equally spaced."""
        return {
            Colour.red: 30.0,
            Colour.yellow: 90.0,
            Colour.green: 150.0,
            Colour.cyan: 210.0,
            Colour.blue: 270.0,
            Colour.magenta: 330.0,
        }[self]

    @property
    def rgb(self) -> tuple[int, int, int]:
        """Perceptually uniform RGB tuple via OKLCH."""
        return _oklch_to_rgb(OKLCH_LIGHTNESS, OKLCH_CHROMA, self.oklch_hue)


class Side(str, Enum):
    left = "left"
    right = "right"


class Condition(BaseModel):
    """A single experimental condition (one row of the factorial design)."""

    crisp_on_top: bool
    crisp_side: Side
    colour_crisp: Colour
    colour_blurred: Colour

    @property
    def correct_answer(self) -> Side:
        """Which side is the foreground (on-top) circle?"""
        if self.crisp_on_top:
            return self.crisp_side
        return Side.right if self.crisp_side == Side.left else Side.left

    @property
    def image_filename(self) -> str:
        top = "crisp-top" if self.crisp_on_top else "blurred-top"
        return f"{top}_{self.crisp_side.value}_{self.colour_crisp.value}_{self.colour_blurred.value}.png"


class TrialResult(BaseModel):
    """Result of a single VLM evaluation trial."""

    condition: Condition
    model: str
    prompt: str
    raw_response: str
    parsed_answer: Side | None
    correct: bool | None
    timestamp: datetime

    @staticmethod
    def now() -> datetime:
        return datetime.now(timezone.utc)


def all_conditions() -> list[Condition]:
    """Generate the full factorial set of conditions, excluding same-colour pairs."""
    conditions = []
    for crisp_on_top in [True, False]:
        for crisp_side in Side:
            for colour_crisp in Colour:
                for colour_blurred in Colour:
                    if colour_crisp == colour_blurred:
                        continue
                    conditions.append(
                        Condition(
                            crisp_on_top=crisp_on_top,
                            crisp_side=crisp_side,
                            colour_crisp=colour_crisp,
                            colour_blurred=colour_blurred,
                        )
                    )
    return conditions
