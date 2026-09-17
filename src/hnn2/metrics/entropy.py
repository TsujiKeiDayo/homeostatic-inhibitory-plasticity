"""Entropy-based concentration score for binned firing rates (bridge metric, C8)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EntropyScores:
    """Entropy and concentration scores for one histogram.

    ``H`` is Shannon entropy in bits; ``Hhat`` is H normalised to [0, 1].
    ``S`` = 1 - Hhat (higher = more concentrated). ``zero_bin_frac`` is the
    mass in the lowest bin; ``S_prime`` = S * (1 - zero_bin_frac).
    """

    H: float
    Hhat: float
    S: float
    zero_bin_frac: float
    S_prime: float


def scores_from_counts(counts: np.ndarray) -> EntropyScores:
    """Score histogram ``counts``, shape ``(n_bins,)``.

    An all-zero histogram yields NaN for ``S`` and ``S_prime`` rather than 1.0,
    since concentration is undefined there.
    """
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total <= 0:
        return EntropyScores(0.0, 0.0, float("nan"), 0.0, float("nan"))

    p = counts / total
    # Empty bins contribute 0 in the limit, but 0*log2(0) is nan, so drop them.
    nonzero = p[p > 0]
    H = float(-np.sum(nonzero * np.log2(nonzero)))

    # log2(n_bins) is the uniform distribution's entropy, the largest H possible.
    n_bins = p.size
    H_max = np.log2(n_bins) if n_bins > 1 else 1.0
    Hhat = H / H_max
    S = 1.0 - Hhat

    # Published bin specs start at [0, 0.05): near-silent, not exactly zero.
    zero_frac = float(p[0])
    return EntropyScores(
        H=H, Hhat=float(Hhat), S=float(S),
        zero_bin_frac=zero_frac, S_prime=float(S * (1.0 - zero_frac)),
    )
