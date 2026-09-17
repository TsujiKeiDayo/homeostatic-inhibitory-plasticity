"""UMAP embedding: declared hyperparameters, seed-tree randomness, determinism.

The embedding is qualitative only (ledger CL6 note; review M8), so nothing here
asserts a scientific ordering — these tests pin the reproducibility properties
the original study lacked (audit I-20: no recorded UMAP settings).
"""

from __future__ import annotations

import numpy as np
import pytest

from hnn2.rng import STREAMS, stream
from hnn2.umap_embed import UmapParams, embed, umap_available

requires_umap = pytest.mark.skipif(
    not umap_available(), reason="optional dependency umap-learn is not installed"
)


def _blobs(n_per_class: int = 40, n_features: int = 12, classes: int = 3) -> np.ndarray:
    rng = np.random.default_rng(0)
    centres = rng.normal(size=(classes, n_features)) * 5.0
    return np.concatenate([
        centres[c] + rng.normal(scale=0.3, size=(n_per_class, n_features))
        for c in range(classes)
    ]).astype(np.float32)


# --- declared settings ------------------------------------------------------

def test_params_defaults_are_declared_and_serialisable():
    params = UmapParams()
    assert params.metric == "cosine"        # matches the CL6 silhouette metric
    assert params.n_components == 2
    assert params.to_json() == {
        "n_neighbors": 15, "min_dist": 0.1, "metric": "cosine", "n_components": 2,
    }


@pytest.mark.parametrize("kwargs", [
    {"n_neighbors": 1},
    {"min_dist": 1.0},
    {"min_dist": -0.1},
    {"n_components": 0},
])
def test_params_reject_out_of_range(kwargs):
    with pytest.raises(ValueError):
        UmapParams(**kwargs)


# --- randomness comes from the seed tree, never a constant ------------------

def test_umap_stream_is_registered_and_append_only():
    # The contract is "append only, never renumber", so a later stream may hold a
    # higher key -- what must never change is the key umap itself was given, and
    # that no two streams collide.
    assert STREAMS["umap"] == 8
    assert len(set(STREAMS.values())) == len(STREAMS)






# --- embedding --------------------------------------------------------------

def test_embed_rejects_bad_input():
    params = UmapParams(n_neighbors=5)
    with pytest.raises(ValueError):
        embed(np.zeros((10,), dtype=np.float32), params, random_state=1)
    with pytest.raises(ValueError):
        embed(np.zeros((4, 3), dtype=np.float32), params, random_state=1)
    bad = _blobs(n_per_class=10)
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        embed(bad, params, random_state=1)


@requires_umap
def test_embed_shape_and_determinism():
    X = _blobs()
    params = UmapParams(n_neighbors=10)
    state = int(stream(0, "umap").integers(0, 2**31))
    first = embed(X, params, random_state=state)
    second = embed(X, params, random_state=state)
    assert first.shape == (X.shape[0], 2)
    assert np.isfinite(first).all()
    np.testing.assert_allclose(first, second)


@requires_umap
def test_embed_varies_with_random_state():
    X = _blobs()
    params = UmapParams(n_neighbors=10)
    a = embed(X, params, random_state=int(stream(0, "umap").integers(0, 2**31)))
    b = embed(X, params, random_state=int(stream(1, "umap").integers(0, 2**31)))
    assert not np.allclose(a, b)
