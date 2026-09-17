import numpy as np
import pytest

from hnn2.data import (
    DatasetParams,
    allocate_proportional_counts,
    dataset_content_hash,
    resize_flat,
    stratified_sample_indices,
)


def _labels(counts: dict[int, int], rng) -> np.ndarray:
    y = np.concatenate([np.full(n, c) for c, n in counts.items()])
    rng.shuffle(y)
    return y


def test_proportional_allocation_sums_and_tracks_proportions():
    rng = np.random.default_rng(0)
    y = _labels({0: 5000, 1: 3000, 2: 2000}, rng)
    alloc = allocate_proportional_counts(y, 1000)
    assert sum(alloc.values()) == 1000
    assert alloc == {0: 500, 1: 300, 2: 200}


def test_allocation_largest_remainder():
    rng = np.random.default_rng(0)
    y = _labels({0: 3, 1: 3, 2: 3}, rng)
    alloc = allocate_proportional_counts(y, 7)
    assert sum(alloc.values()) == 7
    assert set(alloc.values()) <= {2, 3}


def test_allocation_rejects_oversampling():
    with pytest.raises(ValueError):
        allocate_proportional_counts(np.zeros(10, dtype=int), 11)


def test_stratified_sampling_no_replacement_and_deterministic():
    rng = np.random.default_rng(7)
    y = _labels({d: 400 + 13 * d for d in range(10)}, rng)
    idx_a = stratified_sample_indices(y, 1024, np.random.default_rng(42))
    idx_b = stratified_sample_indices(y, 1024, np.random.default_rng(42))
    assert np.array_equal(idx_a, idx_b)
    assert len(np.unique(idx_a)) == 1024

    alloc = allocate_proportional_counts(y, 1024)
    sampled_counts = {int(c): int(n) for c, n in zip(*np.unique(y[idx_a], return_counts=True))}
    assert sampled_counts == alloc


def test_resize_flat_shape_and_range():
    images = np.random.default_rng(0).random((5, 28, 28)).astype(np.float32)
    flat = resize_flat(images, (10, 10))
    assert flat.shape == (5, 100)
    assert flat.dtype == np.float32
    assert float(flat.min()) >= -1e-6 and float(flat.max()) <= 1.0 + 1e-6


def test_content_hash_sensitive_to_data_and_params():
    params_a = DatasetParams((10, 10), (22, 22), 4, 0.9, 0)
    params_b = DatasetParams((10, 10), (22, 22), 4, 0.9, 1)
    arrays = {"x": np.arange(12, dtype=np.float32).reshape(3, 4)}
    assert dataset_content_hash(arrays, params_a) != dataset_content_hash(arrays, params_b)

    mutated = {"x": arrays["x"].copy()}
    mutated["x"][0, 0] += 1e-3
    assert dataset_content_hash(arrays, params_a) != dataset_content_hash(mutated, params_a)
