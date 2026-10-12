from __future__ import annotations

from collections import deque

import numpy as np
from PIL import Image

from perception.dataset.models import Point, Polygon


def instance_pixels(image: Image.Image, instance_value: int) -> set[Point]:
    array = np.asarray(image, dtype=np.uint32)
    packed = array[..., 0] | (array[..., 1] << 8) | (array[..., 2] << 16)
    ys, xs = np.nonzero(packed == instance_value)
    return set(zip(xs.tolist(), ys.tolist()))


def polygons(pixels: set[Point]) -> list[Polygon]:
    remaining = set(pixels)
    output: list[Polygon] = []
    while remaining:
        component = _component(remaining)
        loops = _boundary_loops(component)
        output.extend(loop for loop in loops if len(loop) >= 3)
    return output


def _component(remaining: set[Point]) -> set[Point]:
    start = remaining.pop()
    found = {start}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        for neighbor in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if neighbor in remaining:
                remaining.remove(neighbor)
                found.add(neighbor)
                queue.append(neighbor)
    return found


def _boundary_loops(component: set[Point]) -> list[Polygon]:
    edges: dict[Point, list[Point]] = {}

    def add(start: Point, end: Point) -> None:
        edges.setdefault(start, []).append(end)

    for x, y in component:
        if (x, y - 1) not in component:
            add((x, y), (x + 1, y))
        if (x + 1, y) not in component:
            add((x + 1, y), (x + 1, y + 1))
        if (x, y + 1) not in component:
            add((x + 1, y + 1), (x, y + 1))
        if (x - 1, y) not in component:
            add((x, y + 1), (x, y))

    loops: list[Polygon] = []
    while edges:
        start = next(iter(edges))
        loop = [start]
        current = start
        while True:
            candidates = edges[current]
            following = candidates.pop()
            if not candidates:
                del edges[current]
            if following == start:
                break
            loop.append(following)
            current = following
        simplified = _remove_collinear(loop)
        if len(simplified) >= 3:
            loops.append(tuple(simplified))
    return loops


def _remove_collinear(points: list[Point]) -> list[Point]:
    output = points
    changed = True
    while changed and len(output) > 3:
        changed = False
        kept: list[Point] = []
        for index, point in enumerate(output):
            previous = output[index - 1]
            following = output[(index + 1) % len(output)]
            if _cross(previous, point, following) == 0:
                changed = True
            else:
                kept.append(point)
        output = kept
    return output


def _cross(left: Point, middle: Point, right: Point) -> int:
    return (middle[0] - left[0]) * (right[1] - middle[1]) - (middle[1] - left[1]) * (
        right[0] - middle[0]
    )


def polygon_area(polygon: Polygon) -> float:
    return abs(
        sum(
            left[0] * right[1] - right[0] * left[1]
            for left, right in zip(polygon, polygon[1:] + polygon[:1])
        )
        / 2
    )


def pixel_box(pixels: set[Point]) -> tuple[int, int, int, int]:
    x_values = [point[0] for point in pixels]
    y_values = [point[1] for point in pixels]
    return min(x_values), min(y_values), max(x_values) + 1, max(y_values) + 1
