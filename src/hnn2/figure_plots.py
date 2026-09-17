"""Small drawing and saving helpers; figure assembly lives in the notebooks."""
from pathlib import Path

import numpy as np

from .figure_tables import _source_hashes
from .plots import save_figure
from .result_io import _json


def _points(ax, frame, x, y, *, label, color):
    """Show every finite run value, then the group median (no implicit pooling)."""
    clean = frame[np.isfinite(frame[y])]
    ax.scatter(clean[x], clean[y], color=color, s=14, alpha=.5)
    median = clean.groupby(x)[y].median().sort_index()
    ax.plot(median.index, median.values, color=color, label=label)


def _arm_label(arm, targets):
    model = arm.removesuffix("_trained")
    return f"{arm} (targ={targets[model]:g})" if arm.endswith("_trained") else arm


def _violin_points(ax, values, position, color):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) > 1 and np.ptp(values) > 0:
        body = ax.violinplot(values, positions=[position], showmedians=True)
        for artist in body["bodies"]:
            artist.set_facecolor(color)
    if len(values):
        ax.scatter(position + np.linspace(-.06, .06, len(values)), values, color=color, s=22, zorder=3)
        ax.plot([position-.15, position+.15], [np.median(values)]*2, color=color)


def _histogram_axes(fig, slot, heights):
    """Break only the frequency axis when a large empty gap separates peaks.

    All positive bin heights enter this decision. No bin top lies in the
    omitted interval. Without a clear gap the full linear axis is retained.
    """
    levels = np.unique(np.concatenate(heights))
    levels = levels[levels > 0]
    peak = levels[-1] if len(levels) else 0
    gaps = [(hi-lo, lo, hi) for lo, hi in zip(levels[:-1], levels[1:])
            if hi >= 3*lo and hi-lo >= .15*peak]
    if not gaps:
        ax = fig.add_subplot(slot)
        ax.set_ylim(0, max(.01, peak*1.08))
        return [ax]

    _, lo, hi = max(gaps)
    inner = slot.subgridspec(2, 1, height_ratios=(1, 2), hspace=.08)
    top = fig.add_subplot(inner[0])
    bottom = fig.add_subplot(inner[1], sharex=top)
    top.set_ylim(.9*hi, 1.08*peak)
    bottom.set_ylim(0, 1.1*lo)
    top.spines["bottom"].set_visible(False)
    bottom.spines["top"].set_visible(False)
    top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    for ax, y in ((top, 0), (bottom, 1)):
        for x in (0, 1):
            ax.plot([x-.012, x+.012], [y-.035, y+.035], transform=ax.transAxes,
                    color="black", linewidth=.8, clip_on=False)
    top.text(.98, .05, f"y break: {1.1*lo:.3g} to {.9*hi:.3g}",
             ha="right", va="bottom", transform=top.transAxes, fontsize=6)
    return [top, bottom]


def save_basic_figures(figures, directory, *, png=True, pdf=True, numeric_source=None):
    """Save named figures to a new directory and return their output paths.

    numeric_source must be complete when supplied; its hashes are recorded in
    figure_sources.json. Both format switches off creates no directory.
    """
    if type(png) is not bool or type(pdf) is not bool:
        raise ValueError("png/pdf switches must be boolean")
    if not png and not pdf:
        return []
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"Choose a new figure output directory: {directory}")

    source_hashes = None
    if numeric_source is not None:
        if not (Path(numeric_source) / "COMPLETE").is_file():
            raise ValueError("Figure numeric source is incomplete")
        source_hashes = _source_hashes(numeric_source)

    files = []
    for name, fig in figures.items():
        for extension, enabled in (("png", png), ("pdf", pdf)):
            if enabled:
                files.append(save_figure(fig, directory / f"{name}.{extension}"))
    if files:
        _json(directory / "figure_sources.json", {
            "numeric_source": str(numeric_source) if numeric_source else None,
            "numeric_source_sha256": source_hashes,
            "figures": [p.name for p in files],
            "scope": "basic saved-number plots; no inference or UMAP",
        })
    return files
