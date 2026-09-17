import numpy as np
import pytest

from hnn2.metrics.bins import BinSpec, DroppedMassError, LIFETIME_BINS, POPULATION_BINS
from hnn2.metrics.entropy import scores_from_counts
from hnn2.metrics.silhouette import representation_silhouette


def test_binspec_edges_match_published_convention():
    assert POPULATION_BINS.n_bins == 120
    assert LIFETIME_BINS.n_bins == 60
    assert POPULATION_BINS.edges[0] == 0.0
    assert abs(POPULATION_BINS.edges[-1] - 6.0) < 1e-12


def test_binspec_fail_fast_on_dropped_mass():
    spec = BinSpec(0.0, 1.0, 0.1)
    with pytest.raises(DroppedMassError):
        spec.apply(np.array([0.5, 1.5]))
    counts, dropped = spec.apply(np.array([0.5, 1.5]), allow_dropped=True)
    assert dropped == pytest.approx(0.5)
    assert counts.sum() == 1


def test_entropy_uniform_distribution_gives_zero_S():
    counts = np.ones(64)
    s = scores_from_counts(counts)
    assert s.Hhat == pytest.approx(1.0)
    assert s.S == pytest.approx(0.0)


def test_entropy_single_bin_gives_full_S():
    counts = np.zeros(64)
    counts[3] = 100
    s = scores_from_counts(counts)
    assert s.H == pytest.approx(0.0)
    assert s.S == pytest.approx(1.0)
    assert s.zero_bin_frac == 0.0


def test_entropy_two_equal_bins_analytic():
    counts = np.zeros(16)
    counts[0] = counts[1] = 50
    s = scores_from_counts(counts)
    assert s.H == pytest.approx(1.0)              # 1 bit
    assert s.Hhat == pytest.approx(1.0 / 4.0)     # log2(16) = 4
    assert s.zero_bin_frac == pytest.approx(0.5)
    assert s.S_prime == pytest.approx(s.S * 0.5)


def test_silhouette_separable_clusters_positive():
    rng = np.random.default_rng(0)
    a = rng.normal(loc=(5, 0), scale=0.1, size=(50, 2))
    b = rng.normal(loc=(0, 5), scale=0.1, size=(50, 2))
    X = np.vstack([a, b]).T                       # (features, samples)
    y = np.array([0] * 50 + [1] * 50)
    score = representation_silhouette(X, y)
    assert score is not None and score > 0.9


def test_silhouette_degenerate_returns_none():
    X = np.random.default_rng(0).random((4, 10))
    assert representation_silhouette(X, np.zeros(10, dtype=int)) is None
