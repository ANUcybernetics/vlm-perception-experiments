"""Scotoma: a dual-stream typeface exploiting occlusion edge blur.

Named for the vision-science term for a blind spot in the visual field.
"""

from vlm_perception.scotoma.models import ScotomaStyle, pair_streams
from vlm_perception.scotoma.render import render_scotoma

__all__ = ["ScotomaStyle", "pair_streams", "render_scotoma"]
