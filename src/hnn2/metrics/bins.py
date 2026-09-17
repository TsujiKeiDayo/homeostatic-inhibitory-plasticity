"""Histogram bins with explicit ranges and out-of-range checks (D8; audit I-05)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class DroppedMassError(ValueError):
    """Values fell outside the histogram range and ``allow_dropped`` was False."""

    pass


@dataclass(frozen=True)
class BinSpec:
    """Bins of ``width`` covering ``[lo, hi]``.

    Construction rejects a range that is not a whole number of bins wide.
    """

    lo: float
    hi: float
    width: float

    def __post_init__(self) -> None:
        if not (self.hi > self.lo and self.width > 0):
            raise ValueError(f"invalid BinSpec({self.lo}, {self.hi}, {self.width})")

        # An indivisible range would leave the last edge short of hi, silently
        # binning a narrower range than the spec advertises.
        span = self.hi - self.lo
        if abs(span / self.width - round(span / self.width)) > 1e-9:
            raise ValueError(
                f"BinSpec({self.lo}, {self.hi}, {self.width}): range {span!r} is not "
                f"a whole number of bins wide; the top edge would fall short of hi"
            )

    @property
    def edges(self) -> np.ndarray:
        """Bin boundaries, shape ``(n_bins + 1,)``."""
        # Compute each edge from lo to avoid accumulating rounding error.
        n_bins = int(round((self.hi - self.lo) / self.width))
        return self.lo + self.width * np.arange(n_bins + 1)

    @property
    def n_bins(self) -> int:
        return len(self.edges) - 1

    def apply(
        self,
        values: np.ndarray,
        *,
        allow_dropped: bool = False,
    ) -> tuple[np.ndarray, float]:
        """Return histogram counts and the fraction of values not assigned a bin.

        ``values`` may have any shape; ``counts`` has shape ``(n_bins,)``.
        Empty input returns zero counts and zero dropped mass.

        Raises ``DroppedMassError`` when ``dropped`` would be non-zero unless
        ``allow_dropped=True``.
        """
        values = np.asarray(values, dtype=float).ravel()
        counts, _ = np.histogram(values, self.edges)
        dropped = 1.0 - counts.sum() / values.size if values.size else 0.0

        if dropped > 0 and not allow_dropped:
            # Report the edges actually used, not lo/hi, so the number in the
            # message is the one the values were really compared against.
            edges = self.edges
            raise DroppedMassError(
                f"{dropped:.4%} of mass outside [{edges[0]}, {edges[-1]}] "
                f"(min={values.min():.4g}, max={values.max():.4g}); "
                "pass allow_dropped=True to record instead of fail"
            )
        return counts, float(dropped)


# Fixed ranges inherited from the original study for comparability (C8):
# per-image mean rates use 120 bins; per-unit mean rates use 60 bins. Do not tune.
POPULATION_BINS = BinSpec(0.0, 6.0, 0.05)
LIFETIME_BINS = BinSpec(0.0, 3.0, 0.05)
