from __future__ import annotations

import hashlib

import numpy as np
from PIL import ImageDraw


def colour_for(key: str) -> tuple[int, int, int]:
    digest = hashlib.md5(key.encode()).digest()
    return tuple(90 + d % 166 for d in digest[:3])


def dashed_rectangle(
    draw: ImageDraw.ImageDraw, box, colour, dash: int = 6, width: int = 2
) -> None:
    x0, y0, x1, y1 = box
    for a, b, horizontal, fixed in (
        (x0, x1, True, y0),
        (x0, x1, True, y1),
        (y0, y1, False, x0),
        (y0, y1, False, x1),
    ):
        for start in np.arange(a, b, 2 * dash):
            end = min(start + dash, b)
            draw.line(
                (
                    (start, fixed, end, fixed)
                    if horizontal
                    else (fixed, start, fixed, end)
                ),
                fill=colour,
                width=width,
            )
