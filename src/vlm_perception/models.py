from enum import Enum
from datetime import datetime, timezone

from pydantic import BaseModel


class Colour(str, Enum):
    red = "red"
    yellow = "yellow"
    green = "green"
    cyan = "cyan"
    blue = "blue"
    magenta = "magenta"

    @property
    def hue(self) -> int:
        """HSV hue value (0-360)."""
        return {
            Colour.red: 0,
            Colour.yellow: 60,
            Colour.green: 120,
            Colour.cyan: 180,
            Colour.blue: 240,
            Colour.magenta: 300,
        }[self]

    @property
    def rgb(self) -> tuple[int, int, int]:
        """Full-saturation RGB tuple."""
        import colorsys

        r, g, b = colorsys.hsv_to_rgb(self.hue / 360, 1.0, 1.0)
        return (int(r * 255), int(g * 255), int(b * 255))


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
