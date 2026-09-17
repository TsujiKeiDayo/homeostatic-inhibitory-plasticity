"""Learning-rate lattice and descriptive power-law fits for selected η (SPEC CL8).

Where a variant's best learning rate sits scales with the target rate ρ. The
W_EI update is ∝ η (o_E − ρ) v_I, and v_I is itself ∝ ρ for ``rec`` (I is
driven by E) but ρ-independent for ``ff`` (I is driven by the input); the
``thresh`` update Δθ ∝ η (o_E − ρ) has to shift θ by O(ρ). Counting the update
steps needed to close an O(ρ) gap in a fixed number of epochs gives

    η* ∝ ρ^-2 (rec)      η* ∝ ρ^-1 (ff)      η* ∝ ρ^0 (thresh)

The current experiment uses one fixed grid (``experiment_settings.SWEEP_ETAS``)
for all target cells, rather than centring each grid on the expected scaling.
The fitted exponent is descriptive and carries no acceptance gate
(SPEC.md, section 4).
"""

from __future__ import annotations

import numpy as np

# Exponent b in eta = A * targ ** (-b), from the argument above.
THEORY_EXPONENT: dict[str, float] = {"rec": 2.0, "ff": 1.0, "thresh": 0.0}

# The {1, 3} x 10^n lattice, ascending: the dissertation's own {3e-1, 1e-1,
# 3e-2, ...} spacing (consecutive ratios alternate 3 and 10/3).
LATTICE: tuple[float, ...] = tuple(
    float(f"{mantissa}e{exponent}")
    for exponent in range(-9, 4)
    for mantissa in (1, 3)
)


def lattice_between(low: float, high: float) -> tuple[float, ...]:
    """Every lattice point in ``[low, high]`` (both must be lattice members)."""
    if low not in LATTICE or high not in LATTICE:
        raise ValueError(f"{low} and {high} must both be lattice points {LATTICE}")
    if low >= high:
        raise ValueError("low must be below high")
    return tuple(point for point in LATTICE if low <= point <= high)


def fit_power_law(targs, etas) -> dict[str, float]:
    """Least squares in log-log space: eta = A * targ ** (-b).

    Returns ``A``, ``b`` (positive for a rate that falls with targ), ``r2`` and
    ``n``. Requires at least three paired, positive target and eta values.
    """
    x = np.log10(np.asarray(targs, dtype=float))
    y = np.log10(np.asarray(etas, dtype=float))
    if x.size < 3:
        raise ValueError("fit_power_law needs at least three (targ, eta) points")
    slope, intercept = np.polyfit(x, y, 1)
    r2 = float(np.corrcoef(x, y)[0, 1] ** 2) if x.size > 2 else float("nan")
    return {"A": float(10 ** intercept), "b": float(-slope), "r2": r2, "n": int(x.size)}

