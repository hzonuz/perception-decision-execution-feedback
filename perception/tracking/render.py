from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from perception.contracts import MemoryState, TrackFrame, TrackLifecycle
from perception.drawing import colour_for, dashed_rectangle

STATE_ORDER = (
    MemoryState.VISIBLE,
    MemoryState.PARTIALLY_VISIBLE,
    MemoryState.OCCLUDED,
    MemoryState.OUT_OF_VIEW,
    MemoryState.LOST,
    MemoryState.UNOBSERVABLE,
    MemoryState.TERMINATED,
)
STATE_COLOURS = (
    "#2ca02c",
    "#98df8a",
    "#ff7f0e",
    "#1f77b4",
    "#9a9a9a",
    "#222222",
    "#d62728",
)
EVENT_MARKERS = {
    TrackLifecycle.CREATED: ">",
    TrackLifecycle.REAPPEARED: "o",
    TrackLifecycle.ARCHIVED: "x",
    TrackLifecycle.TERMINATED: "X",
}


def render_tracks(
    frame: TrackFrame, image: np.ndarray | None = None, *, size: int = 384
) -> Image.Image:
    """Seen tracks solid, remembered ones dashed at their last box (out-of-view ones not drawn);
    label: track number, state, existence."""
    tracks = frame.tracks
    width = (
        tracks[0].bounding_box.image_width
        if tracks
        else (image.shape[1] if image is not None else 256)
    )
    height = (
        tracks[0].bounding_box.image_height
        if tracks
        else (image.shape[0] if image is not None else 256)
    )
    if image is None:
        base = Image.new("RGB", (width, height), (150, 170, 190))
        ImageDraw.Draw(base).rectangle(
            (0, height // 2, width, height), fill=(170, 160, 130)
        )
    else:
        base = Image.fromarray(image)
    if not frame.observable:
        base = Image.new("RGB", base.size, (0, 0, 0))
    scale = size / width
    canvas = base.resize((size, int(height * scale)), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=12)
    for track in tracks:
        state = track.memory.state
        if state is MemoryState.OUT_OF_VIEW:
            continue
        b = track.bounding_box
        box = [
            b.x_min * scale,
            b.y_min * scale,
            b.x_max * scale - 1,
            b.y_max * scale - 1,
        ]
        colour = colour_for(track.record.track_id)
        if state in (MemoryState.VISIBLE, MemoryState.PARTIALLY_VISIBLE):
            draw.rectangle(box, outline=colour, width=2)
        else:
            dashed_rectangle(draw, box, colour)
        label = (
            track.semantic_distribution.entries[0].label
            if track.semantic_distribution.entries
            else "?"
        )
        text = (
            f"T{int(track.record.track_id.rsplit('-', 1)[-1])} {label}\n"
            f"{state.value} E={track.memory.existence_probability:.2f}"
        )
        text_box = draw.multiline_textbbox((0, 0), text, font=font, stroke_width=1)
        text_x = max(2, min(box[0] + 2, canvas.width - text_box[2] - 2))
        text_y = max(2, box[1] - text_box[3] - 4)
        draw.multiline_text(
            (text_x, text_y),
            text,
            fill=colour,
            font=font,
            stroke_width=1,
            stroke_fill=(0, 0, 0),
        )
    hidden = [t for t in tracks if t.memory.state is MemoryState.OUT_OF_VIEW]
    footer = (
        f"frame {frame.stamp.frame_index}"
        + ("  UNOBSERVABLE" if not frame.observable else "")
        + (
            f"  out of view: {', '.join('T' + str(int(t.record.track_id.rsplit('-', 1)[-1])) for t in hidden)}"
            if hidden
            else ""
        )
    )
    draw.text(
        (4, canvas.height - 16),
        footer,
        fill=(255, 255, 255),
        font=font,
        stroke_width=2,
        stroke_fill=(0, 0, 0),
    )
    return canvas


def plot_track_timeline(
    frames: Sequence[TrackFrame], *, max_tracks: int = 30, title: str = ""
):
    index = [f.stamp.frame_index for f in frames]
    lifetimes: dict[str, int] = {}
    for f in frames:
        for t in f.tracks:
            lifetimes[t.record.track_id] = lifetimes.get(t.record.track_id, 0) + 1
    keep = sorted(
        sorted(lifetimes, key=lambda k: -lifetimes[k])[:max_tracks],
        key=lambda k: int(k.rsplit("-", 1)[-1]),
    )
    row = {track_id: i for i, track_id in enumerate(keep)}
    grid = np.full((len(keep), len(frames)), np.nan)
    existence = np.full((len(keep), len(frames)), np.nan)
    events = []
    for column, f in enumerate(frames):
        for t in f.tracks:
            if t.record.track_id not in row:
                continue
            r = row[t.record.track_id]
            grid[r, column] = STATE_ORDER.index(t.memory.state)
            existence[r, column] = t.memory.existence_probability
            if t.record.lifecycle in EVENT_MARKERS:
                events.append((column, r, t.record.lifecycle))
    figure, (top, bottom) = plt.subplots(
        2,
        1,
        figsize=(min(18, 4 + 0.08 * len(frames)), 1.2 + 0.32 * len(keep) + 2.6),
        height_ratios=(max(len(keep), 1) * 0.32 + 0.4, 2.4),
        sharex=True,
    )
    top.imshow(
        grid,
        aspect="auto",
        interpolation="nearest",
        cmap=ListedColormap(STATE_COLOURS),
        vmin=-0.5,
        vmax=len(STATE_ORDER) - 0.5,
        extent=(-0.5, len(frames) - 0.5, len(keep) - 0.5, -0.5),
    )
    for column, r, event in events:
        top.plot(
            column,
            r,
            EVENT_MARKERS[event],
            color="white",
            markeredgecolor="black",
            markersize=6,
        )
    top.set_yticks(
        range(len(keep)), [f"T{int(k.rsplit('-', 1)[-1])}" for k in keep], fontsize=8
    )
    top.legend(
        handles=[
            Patch(color=c, label=s.value) for s, c in zip(STATE_ORDER, STATE_COLOURS)
        ]
        + [
            plt.Line2D(
                [],
                [],
                marker=m,
                linestyle="",
                color="white",
                markeredgecolor="black",
                label=e.value,
            )
            for e, m in EVENT_MARKERS.items()
        ],
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
        fontsize=8,
    )
    top.set_title(
        title or f"{frames[0].stamp.episode_id}: track states" if frames else title
    )
    for r, track_id in enumerate(keep):
        bottom.plot(
            range(len(frames)),
            existence[r],
            color=np.array(colour_for(track_id)) / 255,
            linewidth=1.2,
        )
    bottom.set_ylim(0, 1.02)
    bottom.set_ylabel("existence")
    step = max(1, len(frames) // 12)
    bottom.set_xticks(
        range(0, len(frames), step), [index[i] for i in range(0, len(frames), step)]
    )
    bottom.set_xlabel("frame")
    figure.tight_layout()
    return figure
