"""Encoder arms mapping split arrays to readout features (D10, D14, D15).

Every encoder returns ``h_train`` / ``h_val`` / ``h_test`` (plus
``h_test_full`` when requested), each ``(n_samples, n_features)``.
"""

from __future__ import annotations

import numpy as np
import torch

from .config import ExperimentConfig
from .model.params import ModelParams
from .simulate import encode, warm_start_state


def sparse_features(
    params: ModelParams,
    arrays: dict[str, np.ndarray],
    config: ExperimentConfig,
    device: torch.device,
    *,
    include_full_test: bool,
) -> dict[str, np.ndarray]:
    """Settled excitatory output of the Daleian model for each split (V3).

    The warm-start state always comes from the TRAIN split, even when encoding
    val or test, so test data cannot leak into it.
    """
    # (n_samples, n_pixels) on disk -> (n_pixels, n_samples), one column per stimulus.
    x_train_t = torch.as_tensor(arrays["x_train_small"].T, dtype=torch.float32, device=device)
    warm = warm_start_state(params, x_train_t, config)

    # Settle one split from the shared warm state; .T restores (n_samples, n_units).
    def _enc(key: str) -> np.ndarray:
        x = torch.as_tensor(arrays[key].T, dtype=torch.float32, device=device)
        return encode(params, x, config, warm).o_E.T.detach().cpu().numpy().astype(np.float32)

    features = {
        "h_train": _enc("x_train_small"),
        "h_val": _enc("x_val_small"),
        "h_test": _enc("x_test_small"),
    }
    if include_full_test:
        features["h_test_full"] = _enc("x_test_full_small")
    return features


def raw_features(arrays: dict[str, np.ndarray], *, include_full_test: bool) -> dict[str, np.ndarray]:
    """Return the original ``_big`` pixel arrays as readout features.

    Network arms use ``_small`` inputs instead; these arrays are not copied.
    """
    features = {
        "h_train": arrays["x_train_big"],
        "h_val": arrays["x_val_big"],
        "h_test": arrays["x_test_big"],
    }
    if include_full_test:
        features["h_test_full"] = arrays["x_test_full_big"]
    return features


def mlp_frozen_features(
    mlp,
    arrays: dict[str, np.ndarray],
    device: torch.device,
    *,
    include_full_test: bool,
) -> dict[str, np.ndarray]:
    """Extract frozen MLP hidden features from the Daleian arms' ``_small`` inputs.

    Runs without gradient tracking and leaves the model's train/eval mode unchanged.
    """
    # No transpose: torch Linear already expects (batch, features).
    def _enc(key: str) -> np.ndarray:
        x = torch.as_tensor(arrays[key], dtype=torch.float32, device=device)
        with torch.no_grad():
            return mlp.encode(x).detach().cpu().numpy().astype(np.float32)

    features = {
        "h_train": _enc("x_train_small"),
        "h_val": _enc("x_val_small"),
        "h_test": _enc("x_test_small"),
    }
    if include_full_test:
        features["h_test_full"] = _enc("x_test_full_small")
    return features


def labels_dict(arrays: dict[str, np.ndarray], *, include_full_test: bool) -> dict[str, np.ndarray]:
    """Targets row-aligned with the ``h_*`` feature dicts."""
    labels = {
        "y_train": arrays["y_train"],
        "y_val": arrays["y_val"],
        "y_test": arrays["y_test"],
    }
    if include_full_test:
        labels["y_test_full"] = arrays["y_test_full"]
    return labels
