from datetime import UTC, datetime
from enum import StrEnum
from typing import NamedTuple

from pydantic import BaseModel


class ModelSpec(NamedTuple):
    provider: str
    model_id: str


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "claude-opus-4-6": ModelSpec("anthropic", "claude-opus-4-6"),
    "claude-sonnet-4-6": ModelSpec("anthropic", "claude-sonnet-4-6"),
    "claude-haiku-4-5": ModelSpec("anthropic", "claude-haiku-4-5"),
    "gpt-5.4": ModelSpec("openai", "gpt-5.4"),
    "gpt-5.4-mini": ModelSpec("openai", "gpt-5.4-mini"),
    "gpt-5.4-nano": ModelSpec("openai", "gpt-5.4-nano"),
}


def resolve_model(name: str) -> ModelSpec:
    if name in MODEL_REGISTRY:
        return MODEL_REGISTRY[name]
    raise ValueError(
        f"Unknown model: {name!r}. Available: {', '.join(MODEL_REGISTRY)}"
    )

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

    lc = l_ * l_ * l_
    mc = m_ * m_ * m_
    sc = s_ * s_ * s_

    r_lin = +4.0767416621 * lc - 3.3077115913 * mc + 0.2309699292 * sc
    g_lin = -1.2684380046 * lc + 2.6097574011 * mc - 0.3413193965 * sc
    b_lin = -0.0041960863 * lc - 0.7034186147 * mc + 1.7076147010 * sc

    def linear_to_srgb(x: float) -> int:
        v = 12.92 * x if x <= 0.0031308 else 1.055 * x ** (1.0 / 2.4) - 0.055
        return max(0, min(255, round(v * 255)))

    return (linear_to_srgb(r_lin), linear_to_srgb(g_lin), linear_to_srgb(b_lin))


class Colour(StrEnum):
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


class Side(StrEnum):
    left = "left"
    right = "right"


DEFAULT_BLUR_RADIUS = 20
BLUR_SWEEP_RADII = [4, 8, 12, 16, 20]
BLUR_SWEEP_COLOUR_PAIRS: list[tuple[Colour, Colour]] = [
    (Colour.red, Colour.cyan),
    (Colour.yellow, Colour.blue),
    (Colour.green, Colour.magenta),
    (Colour.cyan, Colour.red),
]


class Condition(BaseModel):
    """A single experimental condition (one row of the factorial design)."""

    crisp_on_top: bool
    crisp_side: Side
    colour_crisp: Colour
    colour_blurred: Colour
    blur_radius: int = DEFAULT_BLUR_RADIUS

    @property
    def correct_answer(self) -> Side:
        """Which side is the foreground (on-top) circle?"""
        if self.crisp_on_top:
            return self.crisp_side
        return Side.right if self.crisp_side == Side.left else Side.left

    @property
    def image_filename(self) -> str:
        top = "crisp-top" if self.crisp_on_top else "blurred-top"
        side = self.crisp_side.value
        cc = self.colour_crisp.value
        cb = self.colour_blurred.value
        return f"{top}_{side}_{cc}_{cb}_blur{self.blur_radius}.png"


class TrialResult(BaseModel):
    """Result of a single VLM evaluation trial."""

    condition: Condition
    model: str
    prompt_id: str
    prompt: str
    raw_response: str
    reasoning_trace: str | None = None
    parsed_answer: Side | None
    correct: bool | None
    timestamp: datetime

    @staticmethod
    def now() -> datetime:
        return datetime.now(UTC)


def all_conditions(blur_radius: int = DEFAULT_BLUR_RADIUS) -> list[Condition]:
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
                            blur_radius=blur_radius,
                        )
                    )
    return conditions


def blur_sweep_conditions() -> list[Condition]:
    """Reduced factorial for blur radius sweep.

    2 (depth) x 2 (side) x 4 colour pairs x 5 blur levels = 80 conditions.
    Colour pairs are chosen to span the hue wheel (complementary pairs).
    """
    conditions = []
    for blur_radius in BLUR_SWEEP_RADII:
        for crisp_on_top in [True, False]:
            for crisp_side in Side:
                for colour_crisp, colour_blurred in BLUR_SWEEP_COLOUR_PAIRS:
                    conditions.append(
                        Condition(
                            crisp_on_top=crisp_on_top,
                            crisp_side=crisp_side,
                            colour_crisp=colour_crisp,
                            colour_blurred=colour_blurred,
                            blur_radius=blur_radius,
                        )
                    )
    return conditions
