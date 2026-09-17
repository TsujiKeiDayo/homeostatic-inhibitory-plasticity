"""UMAP embedding as a pure function.

Qualitative only: never cite the output as claim evidence (CL6, review M8).
``umap-learn`` is an optional dependency (``pip install -e .[umap]``); the rest
of the pipeline runs without it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

UMAP_INSTALL_HINT = (
    "umap-learn is not installed - this is an optional dependency. "
    "Install it with:  pip install -e .[umap]"
)


@dataclass(frozen=True)
class UmapParams:
    """Settings for an explicitly requested UMAP fit (SPEC V8)."""

    n_neighbors: int = 15
    min_dist: float = 0.1
    metric: str = "cosine"
    n_components: int = 2

    def __post_init__(self) -> None:
        if self.n_neighbors < 2:
            raise ValueError(f"n_neighbors must be >= 2, got {self.n_neighbors}")
        if not (0.0 <= self.min_dist < 1.0):
            raise ValueError(f"min_dist must be in [0, 1), got {self.min_dist}")
        if self.n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {self.n_components}")

    def to_json(self) -> dict:
        return asdict(self)


def umap_available() -> bool:
    """Return whether the optional umap module can be imported."""
    try:
        import umap  # noqa: F401
    except ImportError:
        return False
    return True


def embed(
    features: np.ndarray,
    params: UmapParams,
    *,
    random_state: int,
) -> np.ndarray:
    """Project ``features`` to ``n_components`` dimensions.

    Input shape is (n_samples, n_features); output is (n_samples, n_components).
    Row *i* of the result corresponds to row *i* of ``features``. Each call fits
    and transforms the given points only — no reusable fitted state remains for
    embedding new points later. ``random_state`` must come from the seed tree
    (``rng.stream(seed, "umap")``), never a hard-coded constant.
    """
    # Validate before importing: a caller passing a malformed matrix should get
    # the same error whether or not the optional dependency is installed.
    X = np.asarray(features, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"features must be 2-D (n_samples, n_features), got {X.shape}")
    if X.shape[0] <= params.n_neighbors:
        raise ValueError(
            f"n_samples ({X.shape[0]}) must exceed n_neighbors ({params.n_neighbors})"
        )
    if not np.isfinite(X).all():
        raise ValueError("features contain non-finite values")

    try:
        import umap
    except ImportError as exc:                       # pragma: no cover - env dependent
        raise ImportError(UMAP_INSTALL_HINT) from exc

    # Reduce the seed-tree draw modulo the 32-bit range accepted by UMAP.
    reducer = umap.UMAP(
        n_neighbors=params.n_neighbors,
        min_dist=params.min_dist,
        metric=params.metric,
        n_components=params.n_components,
        random_state=int(random_state) % (2**32),
    )
    return np.asarray(reducer.fit_transform(X), dtype=np.float32)
