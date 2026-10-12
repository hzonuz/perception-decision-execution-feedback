"""
Render an `SSGSnapshot` over its frame: visible nodes solid, memory-only nodes dashed at their last
seen box, the agent at the bottom centre, key binary relations as labelled arrows, and a side panel
listing nodes and relations (probability, age, stale marker).
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

from perception.contracts import MemoryState, RelationPredicate as P, SSGSnapshot
from perception.drawing import colour_for as _colour, dashed_rectangle as _dashed

DRAWN = (
    P.ON,
    P.OCCLUDES,
    P.LAST_SEEN_BEHIND,
    P.LIKELY_OCCLUDER,
    P.DANGER_BETWEEN,
    P.PATH_BLOCKED_BY,
)
EDGE_COLOURS = {
    P.ON: (60, 180, 75),
    P.OCCLUDES: (145, 30, 180),
    P.LAST_SEEN_BEHIND: (240, 50, 230),
    P.LIKELY_OCCLUDER: (70, 190, 240),
    P.DANGER_BETWEEN: (230, 25, 75),
    P.PATH_BLOCKED_BY: (245, 130, 48),
}
PANEL_SKIP = (
    P.LEFT_OF,
    P.RIGHT_OF,
    P.ABOVE,
    P.BELOW,
)  # directional pairs: n^2 and visible in the image
VISIBLE = (MemoryState.VISIBLE, MemoryState.PARTIALLY_VISIBLE)


def render_snapshot(
    snapshot: SSGSnapshot,
    image: np.ndarray | None = None,
    *,
    size: int = 512,
    drawn: tuple[P, ...] = DRAWN,
    panel_width: int = 420,
    panel_lines: int = 40,
    draw_out_of_view: bool = False,
) -> Image.Image:

    width = (
        snapshot.objects[0].bounding_box.image_width
        if snapshot.objects
        else (image.shape[1] if image is not None else 256)
    )
    height = (
        snapshot.objects[0].bounding_box.image_height
        if snapshot.objects
        else (image.shape[0] if image is not None else 256)
    )
    scale = size / width
    if image is None:
        base = Image.new("RGB", (width, height), (150, 170, 190))
        ImageDraw.Draw(base).rectangle(
            (0, height // 2, width, height), fill=(170, 160, 130)
        )  # floor below the horizon
    else:
        base = Image.fromarray(image)
    canvas = Image.new("RGB", (size + panel_width, int(height * scale)), (24, 24, 28))
    canvas.paste(
        base.resize((size, int(height * scale)), Image.Resampling.NEAREST), (0, 0)
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=12)
    short = {n.node_id: f"o{i}" for i, n in enumerate(snapshot.objects)} | {
        snapshot.agent.node_id: "agent"
    }
    centre = {snapshot.agent.node_id: (size / 2, int(height * scale) - 6)}

    for n in snapshot.objects:
        b = n.bounding_box
        box = [
            b.x_min * scale,
            b.y_min * scale,
            b.x_max * scale - 1,
            b.y_max * scale - 1,
        ]
        colour = _colour(
            n.semantic_distribution.entries[0].label
            if n.semantic_distribution.entries
            else "?"
        )
        centre[n.node_id] = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        if n.memory_state is MemoryState.OUT_OF_VIEW and not draw_out_of_view:
            continue
        label = (
            n.semantic_distribution.entries[0].label
            if n.semantic_distribution.entries
            else "unknown"
        )
        if n.memory_state in VISIBLE:
            draw.rectangle(box, outline=colour, width=2)
            text = f"{short[n.node_id]} {label}"
        else:
            _dashed(draw, box, colour)
            text = f"{short[n.node_id]} {label} [{n.memory_state.value}]"
            if n.memory_state is MemoryState.TERMINATED:
                draw.line(box, fill=colour, width=2)
                draw.line((box[0], box[3], box[2], box[1]), fill=colour, width=2)
        draw.text(
            (box[0] + 2, max(box[1] - 14, 0)),
            text,
            fill=colour,
            font=font,
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
    ax, ay = centre[snapshot.agent.node_id]
    draw.polygon(
        [(ax, ay - 12), (ax - 9, ay + 4), (ax + 9, ay + 4)], fill=(255, 255, 255)
    )

    for r in snapshot.relations:
        if r.predicate not in drawn or r.object_node_id not in centre:
            continue
        (x0, y0), (x1, y1) = centre[r.subject_node_id], centre[r.object_node_id]
        colour = EDGE_COLOURS.get(r.predicate, (255, 255, 255))
        draw.line((x0, y0, x1, y1), fill=colour, width=2)
        _arrowhead(draw, (x0, y0), (x1, y1), colour)
        draw.text(
            ((x0 + x1) / 2 + 3, (y0 + y1) / 2),
            f"{r.predicate.value} {r.probability:.2f}",
            fill=colour,
            font=font,
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )

    lines = [
        (
            f"{snapshot.graph_id}{'  [macro]' if snapshot.is_macro_step else ''}",
            (255, 255, 255),
        ),
        (
            f"nodes {len(snapshot.objects)}  relations {len(snapshot.relations)}  confidence {snapshot.confidence:.2f}",
            (200, 200, 200),
        ),
    ]
    for n in snapshot.objects:
        observed = n.memory_state in VISIBLE
        label = (
            n.semantic_distribution.entries[0]
            if n.semantic_distribution.entries
            else None
        )
        distance = (
            f" d={n.relative_distance:.1f}" if n.relative_distance is not None else ""
        )
        lines.append(
            (
                f"{short[n.node_id]:>3} {label.label if label else '?'} {label.probability if label else 0:.2f}"
                f"  {n.memory_state.value}  E={n.existence_probability:.2f}{distance}",
                (230, 230, 230) if observed else (150, 170, 255),
            )
        )
    lines.append(("", (0, 0, 0)))
    for r in sorted(snapshot.relations, key=lambda r: -r.probability):
        if r.predicate in PANEL_SKIP:
            continue
        target = (
            short.get(r.object_node_id, r.object_node_id)
            if r.object_node_id
            else repr(r.symbolic_value)
        )
        stale = (
            " stale" if any(e.startswith("stale_frames=") for e in r.evidence) else ""
        )
        lines.append(
            (
                f"{r.predicate.value}({short.get(r.subject_node_id, '?')},{target}) {r.probability:.2f} age {r.age_frames}{stale}",
                EDGE_COLOURS.get(r.predicate, (190, 190, 190)),
            )
        )
    directional = sum(r.predicate in PANEL_SKIP for r in snapshot.relations)
    if directional:
        lines.append((f"+ {directional} left_of/right_of/above/below", (140, 140, 140)))
    for i, (text, colour) in enumerate(lines[:panel_lines]):
        draw.text((size + 10, 8 + 15 * i), text, fill=colour, font=font)
    if len(lines) > panel_lines:
        draw.text(
            (size + 10, 8 + 15 * panel_lines),
            f"... {len(lines) - panel_lines} more",
            fill=(140, 140, 140),
            font=font,
        )
    return canvas


def _arrowhead(draw, start, end, colour, length: float = 9.0) -> None:
    direction = np.subtract(end, start)
    norm = float(np.linalg.norm(direction))
    if norm < 1:
        return
    u = direction / norm
    left, right = np.array([-u[1], u[0]]), np.array([u[1], -u[0]])
    tip = np.array(end)
    draw.polygon(
        [
            tuple(tip),
            tuple(tip - length * u + 0.5 * length * left),
            tuple(tip - length * u + 0.5 * length * right),
        ],
        fill=colour,
    )


def draw_graph(snapshot: SSGSnapshot, *, skip: tuple[P, ...] = PANEL_SKIP, ax=None):
    """
    Node-link diagram laid out like the agent's view: x = image position, y = distance
    (floor-contact estimate, or image row when unknown); out-of-view nodes in a column on the
    right. Visible nodes filled, memory-only nodes hollow; binary edges labelled with probability,
    unary ones listed under their node. Needs matplotlib and networkx.
    """
    import matplotlib.pyplot as plt
    import networkx as nx

    graph = nx.MultiDiGraph()
    width = snapshot.objects[0].bounding_box.image_width if snapshot.objects else 256
    height = snapshot.objects[0].bounding_box.image_height if snapshot.objects else 256
    positions, labels, colours, filled = {}, {}, {}, {}
    agent = snapshot.agent.node_id
    graph.add_node(agent)
    positions[agent], labels[agent], colours[agent], filled[agent] = (
        (width / 2, 0.0),
        "agent",
        "#444444",
        True,
    )
    out_of_view = 0
    for i, n in enumerate(snapshot.objects):
        graph.add_node(n.node_id)
        b = n.bounding_box
        label = (
            n.semantic_distribution.entries[0].label
            if n.semantic_distribution.entries
            else "?"
        )
        if n.memory_state is MemoryState.OUT_OF_VIEW:
            positions[n.node_id] = (width * 1.15, 2.0 + 2.2 * out_of_view)
            out_of_view += 1
        else:
            depth = (
                n.relative_distance
                if n.relative_distance is not None
                else 25.0 * (1 - b.y_max / height) + 1
            )
            positions[n.node_id] = ((b.x_min + b.x_max) / 2, min(depth, 30.0))
        labels[n.node_id] = f"o{i} {label}\n{n.memory_state.value}"
        colours[n.node_id] = "#%02x%02x%02x" % _colour(label)
        filled[n.node_id] = n.memory_state in VISIBLE
    unary: dict[str, list[str]] = {}
    for r in snapshot.relations:
        if r.predicate in skip:
            continue
        if r.object_node_id is None:
            unary.setdefault(r.subject_node_id, []).append(
                f"{r.predicate.value}={r.symbolic_value!r} {r.probability:.2f}"
            )
        else:
            graph.add_edge(
                r.subject_node_id,
                r.object_node_id,
                label=f"{r.predicate.value} {r.probability:.2f}",
                stale=any(e.startswith("stale_frames=") for e in r.evidence),
            )
    if ax is None:
        _, ax = plt.subplots(figsize=(13, 8))
    for node in graph.nodes:
        x, y = positions[node]
        ax.scatter(
            [x],
            [y],
            s=900,
            facecolors=colours[node] if filled[node] else "white",
            edgecolors=colours[node],
            linewidths=2.5,
            linestyle="-" if filled[node] else "--",
            zorder=3,
        )
        ax.text(x, y - 1.1, labels[node], ha="center", va="top", fontsize=8, zorder=4)
        if node in unary:
            ax.text(
                x,
                y + 1.0,
                "\n".join(unary[node][:4]),
                ha="center",
                va="bottom",
                fontsize=6.5,
                color="#555555",
                zorder=4,
            )
    grouped: dict[tuple[str, str], list[tuple[str, bool]]] = {}
    for s, t, data in graph.edges(data=True):
        grouped.setdefault((s, t), []).append((data["label"], data["stale"]))
    span = np.ptp([p[0] for p in positions.values()] or [1]) or 1.0
    for (s, t), items in grouped.items():  # one arc per direction, its labels stacked
        (x0, y0), (x1, y1) = positions[s], positions[t]
        all_stale = all(stale for _, stale in items)
        colour = "#999999" if all_stale else "#1f4e79"
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            zorder=2,
            arrowprops=dict(
                arrowstyle="-|>",
                color=colour,
                lw=1.3,
                linestyle="--" if all_stale else "-",
                connectionstyle="arc3,rad=0.25",
                shrinkA=16,
                shrinkB=16,
            ),
        )
        # label beside the arc's midpoint, on the side the arc bulges to
        dx, dy = (x1 - x0) / span, (y1 - y0) / 30.0
        norm = max((dx * dx + dy * dy) ** 0.5, 1e-6)
        mx, my = (x0 + x1) / 2 + 0.18 * (dy / norm) * span * 0.25, (
            y0 + y1
        ) / 2 - 0.18 * (dx / norm) * 30.0 * 0.25
        text = "\n".join(
            f"{label}{' (stale)' if stale else ''}" for label, stale in items
        )
        ax.text(
            mx,
            my,
            text,
            fontsize=6.5,
            color=colour,
            ha="center",
            va="center",
            zorder=4,
            bbox=dict(
                boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8
            ),
        )
    ax.set_xlim(-10, width * 1.3)
    ax.set_ylim(-3, max([p[1] for p in positions.values()] + [10]) + 4)
    ax.set_xlabel("image x (left <-> right)")
    ax.set_ylabel("distance from agent (floor contact)")
    ax.set_title(
        f"{snapshot.graph_id}: {len(snapshot.objects)} nodes, {graph.number_of_edges()} binary edges "
        f"(dashed = stale, hollow = memory-only; out-of-view column on the right)",
        fontsize=9,
    )
    return ax


def plot_relation_timeline(
    snapshots,
    *,
    predicates: tuple[P, ...] | None = None,
    skip: tuple[P, ...] = PANEL_SKIP,
    max_rows: int = 40,
    title: str = "",
):

    short = {}
    for s in snapshots:
        for i, n in enumerate(s.objects):
            label = (
                n.semantic_distribution.entries[0].label
                if n.semantic_distribution.entries
                else "?"
            )
            short.setdefault(
                n.node_id, f"T{int(n.track_id.rsplit('-', 1)[-1])}:{label}"
            )
    short["agent"] = "agent"
    rows: dict[tuple, int] = {}
    cells = []
    for column, s in enumerate(snapshots):
        for r in s.relations:
            if (
                predicates is not None and r.predicate not in predicates
            ) or r.predicate in skip:
                continue
            key = (r.predicate.value, r.subject_node_id, r.object_node_id or "")
            rows.setdefault(key, len(rows))
            cells.append(
                (
                    rows[key],
                    column,
                    r.probability,
                    any(e.startswith("stale_frames=") for e in r.evidence),
                )
            )
    order = sorted(rows, key=lambda k: (k[0], k[1], k[2]))[:max_rows]
    remap = {key: i for i, key in enumerate(order)}
    grid = np.full((len(order), len(snapshots)), np.nan)
    stale = []
    for row, column, p, is_stale in cells:
        key = next(k for k, v in rows.items() if v == row)
        if key in remap:
            grid[remap[key], column] = p
            if is_stale:
                stale.append((column, remap[key]))
    figure, ax = plt.subplots(
        figsize=(min(18, 4 + 0.2 * len(snapshots)), 1 + 0.28 * max(len(order), 1))
    )
    image = ax.imshow(
        grid, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0, vmax=1
    )
    if stale:
        ax.plot(*zip(*stale), ".", color="white", markersize=3)
    ax.set_yticks(
        range(len(order)),
        [
            f"{p}({short.get(s, s)}, {short.get(o, o) if o else 'value'})"
            for p, s, o in order
        ],
        fontsize=7,
    )
    step = max(1, len(snapshots) // 15)
    ax.set_xticks(
        range(0, len(snapshots), step),
        [snapshots[i].stamp.frame_index for i in range(0, len(snapshots), step)],
    )
    ax.set_xlabel("frame (white dot = stale, carried without fresh evidence)")
    ax.set_title(title or "relation probability over time")
    figure.colorbar(image, ax=ax, fraction=0.02, pad=0.01, label="probability")
    figure.tight_layout()
    return figure
