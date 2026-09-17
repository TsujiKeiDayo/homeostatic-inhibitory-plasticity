"""Settling-loop execution: warm start and batched settling (C2/C5; D2).

Adaptation builds a warm state from the train split at each epoch's start
and reuses it across that epoch's batches while parameters change. Probes
rebuild it with the post-epoch parameters, still from train even when
measuring val or test (V3).

Every function also accepts stacked params (``ModelParams.stack``, SPEC V12):
states carry a leading run axis. Inputs may be shared (nIn, B) arrays or
per-run (R, nIn, B) arrays, with samples in the last axis.
"""

from __future__ import annotations

import torch

from .config import ExperimentConfig
from .model.dynamics import Decays, input_drive, step
from .model.params import ModelParams, State, zero_state


def settle(
    params: ModelParams,
    x: torch.Tensor,
    config: ExperimentConfig,
    initial: State | None = None,
) -> State:
    """Run the settling map on ``x`` shaped (nIn, B), or (R, nIn, B) for R runs.

    Always runs the full ``config.settle_steps`` updates (V1) — never stops
    early on convergence. ``initial`` is copied, not mutated.
    """
    decays = Decays.from_taus(config.tau_e, config.tau_i)
    state = (
        initial.clone()
        if initial is not None
        else zero_state(params, x.shape[-1], x.device)
    )
    # The stimulus is constant throughout settling, so reuse its input current.
    drive = input_drive(params, x)

    # The homeostatic rule needs settled values, not gradients through the loop.
    with torch.no_grad():
        for _ in range(int(config.settle_steps)):
            state = step(params, state, drive, decays, x)
    return state


def warm_start_state(
    params: ModelParams,
    x_train: torch.Tensor,
    config: ExperimentConfig,
) -> State:
    """Stabilised initial state from the first ``n_stabilise`` train samples (C5, V3).

    Deterministic: the FIRST samples in split order, not a random draw,
    presented sequentially with state carried across them. Returns a single
    (units, 1) state, with a leading run axis for stacked networks.
    """
    # Guard for the tiny splits used in tests.
    count = min(int(config.n_stabilise), int(x_train.shape[-1]))
    state = zero_state(params, 1, x_train.device)
    for i in range(count):
        # i:i+1 keeps a batch of one; initial=state chains the presentations.
        state = settle(params, x_train[..., i : i + 1], config, initial=state)
    return state


def broadcast_state(state: State, batch: int) -> State:
    """Expand a single-column state to ``batch`` identical contiguous columns."""
    def _wide(tensor: torch.Tensor) -> torch.Tensor:
        # Only the last (sample) axis widens; a leading run axis is kept as is.
        return tensor.expand(*tensor.shape[:-1], batch).contiguous()

    return State(v_E=_wide(state.v_E), o_E=_wide(state.o_E), v_I=_wide(state.v_I))


def encode(
    params: ModelParams,
    x: torch.Tensor,
    config: ExperimentConfig,
    warm: State,
) -> State:
    """Settled response of batch ``x``, every sample starting from the shared warm state."""
    return settle(params, x, config, initial=broadcast_state(warm, x.shape[-1]))
