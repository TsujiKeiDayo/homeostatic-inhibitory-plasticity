"""Class separability of a representation, scored by silhouette (bridge metric, C8)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import silhouette_score


def representation_silhouette(
    o_E: np.ndarray,
    labels: np.ndarray,
    *,
    metric: str = "cosine",
    center: bool = False,
) -> float | None:
    """Score class separation in ``o_E`` of shape (n_units, n_samples).

    ``labels`` has shape (n_samples,). Scores range from -1 to 1, with higher
    values meaning tighter class clusters relative to other classes.
    ``center=True`` subtracts the mean feature vector first (M7).

    Returns ``None`` — not a default value, and not the NaN the entropy scores
    use — when the score is undefined: fewer than 2 samples, fewer than 2
    classes, one class per sample, or non-finite input.
    """
    # sklearn wants (n_samples, n_features); the simulator produces the transpose.
    X = np.asarray(o_E, dtype=np.float64).T
    y = np.asarray(labels, dtype=int)
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"samples mismatch: X {X.shape} vs labels {y.shape}")
    if not 2 <= np.unique(y).size < X.shape[0] or not np.isfinite(X).all():
        return None

    # axis=0 averages over samples, so this removes each unit's baseline activity.
    if center:
        X = X - X.mean(axis=0, keepdims=True)
    return float(silhouette_score(X, y, metric=metric))
