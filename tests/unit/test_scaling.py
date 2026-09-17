import numpy as np
import pytest

from hnn2.hp.scaling import LATTICE, THEORY_EXPONENT, fit_power_law, lattice_between


def test_lattice_is_ascending_one_three_pattern():
    assert all(a < b for a, b in zip(LATTICE, LATTICE[1:]))
    ratios = {round(b / a, 3) for a, b in zip(LATTICE, LATTICE[1:])}
    assert ratios == {3.0, round(10 / 3, 3)}


def test_lattice_between_is_inclusive_and_on_lattice():
    grid = lattice_between(1e-6, 30.0)
    assert grid[0] == 1e-6 and grid[-1] == 30.0
    assert len(grid) == 16
    assert all(g in LATTICE for g in grid)
    assert {1e-3, 1e-4, 0.1} <= set(grid)     # the dissertation's reported etas


@pytest.mark.parametrize("low, high", [(2e-6, 30.0), (1e-6, 25.0), (1.0, 1e-3)])
def test_lattice_between_rejects_off_lattice_or_inverted_bounds(low, high):
    with pytest.raises(ValueError):
        lattice_between(low, high)


def test_fit_power_law_recovers_a_planted_exponent():
    targs = np.array([0.05, 0.1, 0.2, 0.4, 0.8, 1.6])
    etas = 2e-4 * targs ** (-2.0)
    fit = fit_power_law(targs, etas)
    assert fit["b"] == pytest.approx(2.0)
    assert fit["A"] == pytest.approx(2e-4)
    assert fit["r2"] == pytest.approx(1.0)
    assert fit["n"] == 6


def test_fit_power_law_needs_three_points():
    with pytest.raises(ValueError):
        fit_power_law([0.1, 0.2], [1.0, 0.5])


def test_theory_exponents_cover_every_model():
    assert set(THEORY_EXPONENT) == {"rec", "ff", "thresh"}
