"""Shared model colours and labels, with explicit figure saving."""

from pathlib import Path

MODEL_COLORS = {"rec": "#1f77b4", "ff": "#ff7f0e", "thresh": "#d62728"}
MODEL_LABELS = {"rec": "Recurrent", "ff": "Feed-forward", "thresh": "Threshold"}
BASELINE_COLORS = {"raw": "#7f7f7f", "mlp": "#2ca02c"}


def save_figure(fig, path, *, dpi=150, overwrite=False):
    """Save a composed figure, leaving it open for interactive display.

    No data reads or scientific recomputation. Record the numerical input and
    analysis settings in the calling script; choose labels and format there.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path
