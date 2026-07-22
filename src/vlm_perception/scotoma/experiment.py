"""Experimental design for the Scotoma VLM transcription experiment.

Tests whether VLMs read the crisp "robot" stream of a Scotoma render while
a human would read the blurred "real" stream. The string pools are
space-free, length-matched, uppercase, with a minimum pairwise positional
Hamming distance of 8/10 within each set so the bias index is not
attenuated by similar pairs. The pseudoword set controls for the language
prior.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from vlm_perception.models import Colour
from vlm_perception.scotoma.models import ScotomaStyle

STRING_LENGTH = 10
MIN_POOL_HAMMING = 8

# Selected by clique search over common 10-letter words (pairwise
# positional Hamming >= 8 within each set; see tests).
ENGLISH_POOL = [
    "ABSOLUTELY",
    "BACKGROUND",
    "COLLECTION",
    "COMMITMENT",
    "DIFFERENCE",
    "EVERYTHING",
    "HISTORICAL",
    "INDIVIDUAL",
]

# Pronounceable CV(C) pseudowords generated with a fixed seed, filtered
# for pronounceability, same Hamming constraint.
PSEUDO_POOL = [
    "BRAFLATOML",
    "MOFIMFUSTU",
    "SNASMILBUL",
    "KLEKLOSKAT",
    "GRIMPUNVUT",
    "FLERKAFLES",
    "DUSNORVIKU",
    "NEPLASTEKT",
]

POOLS: dict[str, list[str]] = {"english": ENGLISH_POOL, "pseudo": PSEUDO_POOL}

# 0 = no-blur baseline; 0.07 is the tuned Scotoma default. blur_px =
# blur_fraction * font_size.
BLUR_FRACTIONS = [0.0, 0.02, 0.04, 0.07, 0.10, 0.14]

DEFAULT_OFFSET_FRACTION = 0.38
DEFAULT_FONT_SIZE = 96

PoolName = Literal["english", "pseudo"]


def hamming(a: str, b: str) -> int:
    """Positional Hamming distance between two equal-length strings."""
    if len(a) != len(b):
        raise ValueError(f"Equal-length strings required: {len(a)} vs {len(b)}")
    return sum(x != y for x, y in zip(a, b, strict=True))


class ScotomaCondition(BaseModel):
    """One cell of the Scotoma factorial design (or a solo legibility trial)."""

    string_real: str
    # Empty robot stream marks a solo legibility-baseline trial.
    string_robot: str = ""
    pool: PoolName
    blur_fraction: float = Field(ge=0.0)
    blurred_on_top: bool = True
    colour_real: Colour = Colour.red
    colour_robot: Colour = Colour.cyan
    offset_fraction: float = DEFAULT_OFFSET_FRACTION
    font_size: int = Field(default=DEFAULT_FONT_SIZE, gt=0)

    @property
    def is_solo(self) -> bool:
        return self.string_robot == ""

    @property
    def blur_px(self) -> float:
        return self.blur_fraction * self.font_size

    @property
    def d_pair(self) -> int | None:
        """Positional Hamming distance between the two streams (covariate)."""
        if self.is_solo:
            return None
        return hamming(self.string_real, self.string_robot)

    def style(self) -> ScotomaStyle:
        return ScotomaStyle(
            font_size=self.font_size,
            blur_fraction=self.blur_fraction,
            offset_fraction=self.offset_fraction if not self.is_solo else 0.0,
            colour_real=self.colour_real,
            colour_robot=self.colour_robot,
            blurred_on_top=self.blurred_on_top,
        )

    @property
    def image_filename(self) -> str:
        if self.is_solo:
            return (
                f"solo_{self.string_real}_{self.colour_real.value}"
                f"_f{self.font_size}.png"
            )
        depth = "blurredtop" if self.blurred_on_top else "crisptop"
        return (
            f"{self.string_real}-{self.string_robot}"
            f"_blur{self.blur_fraction:g}_{depth}"
            f"_{self.colour_real.value}_f{self.font_size}.png"
        )


class ScotomaTrialResult(BaseModel):
    """A single transcription trial, scored against both streams."""

    condition: ScotomaCondition
    model: str
    prompt_id: str
    prompt: str
    raw_response: str
    reasoning_trace: str | None = None
    # Normalised parsed transcription(s); multiple candidates joined by "|".
    raw_transcription: str | None
    dist_real_lev: float | None
    dist_robot_lev: float | None
    dist_real_ham: float | None
    dist_robot_ham: float | None
    bias_index_lev: float | None
    bias_index_ham: float | None
    timestamp: datetime

    @staticmethod
    def now() -> datetime:
        return datetime.now(UTC)


def ordered_pairs(pool: list[str]) -> list[tuple[str, str, Colour]]:
    """Partition a pool of 8 into 4 disjoint pairs; return 8 ordered pairs.

    Each pair is rendered in both role orderings (reciprocal diptych
    logic) so every string appears once as real and once as robot.
    Colour-role assignment is counterbalanced across ordered pairs (half
    red-real, half cyan-real) rather than crossed as a factor, removing
    the blur-hue confound at zero extra trials. Returns tuples of
    (string_real, string_robot, colour_real).
    """
    if len(pool) != 8:
        raise ValueError(f"Pool must have 8 strings, got {len(pool)}")
    out: list[tuple[str, str, Colour]] = []
    for pair_idx in range(4):
        a, b = pool[2 * pair_idx], pool[2 * pair_idx + 1]
        first, second = (
            (Colour.red, Colour.cyan)
            if pair_idx % 2 == 0
            else (Colour.cyan, Colour.red)
        )
        out.append((a, b, first))
        out.append((b, a, second))
    return out


def scotoma_conditions(font_size: int = DEFAULT_FONT_SIZE) -> list[ScotomaCondition]:
    """The main sweep: 16 ordered pairs x 6 blur levels x 2 depth orders = 192."""
    conditions = []
    for pool_name, pool in POOLS.items():
        for real, robot, colour_real in ordered_pairs(pool):
            colour_robot = Colour.cyan if colour_real == Colour.red else Colour.red
            for blur_fraction in BLUR_FRACTIONS:
                for blurred_on_top in [True, False]:
                    conditions.append(
                        ScotomaCondition(
                            string_real=real,
                            string_robot=robot,
                            pool=pool_name,
                            blur_fraction=blur_fraction,
                            blurred_on_top=blurred_on_top,
                            colour_real=colour_real,
                            colour_robot=colour_robot,
                            font_size=font_size,
                        )
                    )
    return conditions


def generate_stimuli(
    output_dir: Path, conditions: list[ScotomaCondition]
) -> list[Path]:
    """Render every condition's stimulus image into output_dir."""
    from vlm_perception.scotoma.render import render_scotoma, render_solo

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for c in conditions:
        if c.is_solo:
            img = render_solo(c.string_real, c.style())
        else:
            img = render_scotoma(c.string_real, c.string_robot, c.style())
        path = output_dir / c.image_filename
        img.save(path)
        paths.append(path)
    return paths


def legibility_conditions(font_size: int = DEFAULT_FONT_SIZE) -> list[ScotomaCondition]:
    """Solo unblurred renders of every pool string (legibility baseline).

    Colour alternates across pool positions so both stream colours are
    covered without doubling the trial count.
    """
    conditions = []
    for pool_name, pool in POOLS.items():
        for i, s in enumerate(pool):
            colour = Colour.red if i % 2 == 0 else Colour.cyan
            conditions.append(
                ScotomaCondition(
                    string_real=s,
                    string_robot="",
                    pool=pool_name,
                    blur_fraction=0.0,
                    colour_real=colour,
                    font_size=font_size,
                )
            )
    return conditions
