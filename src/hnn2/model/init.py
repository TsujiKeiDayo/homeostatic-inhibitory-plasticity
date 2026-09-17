"""Weight initialisation (C4; D3, V2).

Random weights are drawn from Exp(1), so every weight is non-negative.
Formulas: SPEC.md section 3.
"""

from __future__ import annotations

import math

import torch

from ..config import ExperimentConfig
from ..rng import stream
from .params import ModelParams


def build_params(
    model_code: str,
    config: ExperimentConfig,
    seed_index: int,
    n_input: int,
    device: torch.device,
) -> ModelParams:
    """Draw the initial weights for one run.

    Deterministic: the same arguments always yield bit-identical weights.
    ``rec``/``ff`` raise NotImplementedError unless ``config.n_inhibitory`` is 1.
    """
    nE, nI = config.n_excitatory, config.n_inhibitory

    # Input -> E weights, (nE, n_input); mu below reuses these exact float64 draws.
    w_EIn_np = stream(seed_index, "init/w_EIn").exponential(
        scale=1.0, size=(nE, n_input)
    ) / math.sqrt(n_input)
    w_EIn = torch.as_tensor(w_EIn_np, dtype=torch.float32, device=device)
    theta = torch.full((nE,), float(config.theta_init), dtype=torch.float32, device=device)

    # thresh has no I population: input weights plus thresholds are all it needs.
    if model_code == "thresh":
        return ModelParams(model_code=model_code, w_EIn=w_EIn, theta=theta)

    # mu_i = how strongly E unit i is driven; both E<->I projections reuse it.
    mu = w_EIn_np.mean(axis=1, keepdims=True)  # (nE, 1)
    # I -> E: share of the inhibitory potential each E unit receives. (nE, nI).
    w_EI = torch.as_tensor(mu / math.sqrt(nI), dtype=torch.float32, device=device)
    # mu is (nE, 1), so sqrt(nI) rescales but never widens: a config asking for
    # more I units would silently run as one.
    if w_EI.shape[1] != nI:
        raise NotImplementedError(
            f"n_inhibitory={nI} requested but the initialisation scheme builds "
            f"w_EI with {w_EI.shape[1]} inhibitory unit(s); multi-unit I is not "
            f"implemented"
        )

    if model_code == "rec":
        # E -> I: the I unit reads out the E population using that same profile.
        w_IE = torch.as_tensor(mu.T / math.sqrt(nE), dtype=torch.float32, device=device)
        return ModelParams(model_code=model_code, w_EIn=w_EIn, theta=theta, w_EI=w_EI, w_IE=w_IE)

    if model_code == "ff":
        # Input -> I, (nI, n_input): its own stream, so I sees its own random
        # view of the stimulus rather than a copy of E's.
        w_IIn_np = stream(seed_index, "init/w_IIn").exponential(
            scale=1.0, size=(nI, n_input)
        ) / math.sqrt(n_input)
        w_IIn = torch.as_tensor(w_IIn_np, dtype=torch.float32, device=device)
        return ModelParams(model_code=model_code, w_EIn=w_EIn, theta=theta, w_EI=w_EI, w_IIn=w_IIn)

    raise ValueError(f"unknown model_code: {model_code!r}")
