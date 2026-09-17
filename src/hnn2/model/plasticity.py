"""Homeostatic update rules as pure functions (C3; D13).

  rec/ff:  W_EI <- clamp(W_EI + eta * (o_E - targ) @ pre_signal^T / B, min=0)
  thresh:  theta <- clamp(theta + eta * mean_B(o_E - targ), min=0)

Not supervised learning: there are no labels and no loss is differentiated;
the only error signal is ``o_E - targ``. Raising W_EI means *more* suppression,
since inhibition is subtracted in ``dynamics.step``. Deviations from the
textbook rule: SPEC.md section 2.
"""

from __future__ import annotations

import torch

from .params import ModelParams


def _per_run(value, ndim: int):
    """Broadcast a per-run vector along the leading axis of an ``ndim``-D tensor.

    Scalars and 0-D tensors pass through unchanged (SPEC V12).
    """
    if isinstance(value, torch.Tensor) and value.dim() == 1:
        return value.view(value.shape[0], *([1] * (ndim - 1)))
    return value


def update(
    params: ModelParams,
    o_E: torch.Tensor,          # (nE, B) settled excitatory output
    pre_signal: torch.Tensor | None,  # (nI, B) presynaptic factor; None for thresh
    targ: float | torch.Tensor,       # scalar, or (R,) per stacked network
    eta: float | torch.Tensor,        # scalar, or (R,) per stacked network
) -> ModelParams:
    """Apply one homeostatic update without mutating ``params``.

    Stacked params (SPEC V12) prepend a run axis R to every tensor;
    ``targ`` and ``eta`` may then be ``(R,)`` vectors.
    """
    # (nE, B) postsynaptic error: how far each unit is above/below target.
    error = o_E - _per_run(targ, o_E.dim())

    if params.model_code in ("rec", "ff"):
        if pre_signal is None:
            raise ValueError("rec/ff update requires a pre_signal")
        batch = float(o_E.shape[-1])
        # Batch-averaged pre-post correlation, (nE, B) @ (B, nI) -> (nE, nI).
        delta = (error @ pre_signal.transpose(-1, -2)) / batch
        # Step, then clamp — an inhibitory weight may reach zero but not cross it.
        new_w_EI = torch.clamp(params.w_EI + _per_run(eta, delta.dim()) * delta, min=0.0)
        return params.with_(w_EI=new_w_EI)

    if params.model_code == "thresh":
        # No presynaptic partner: average the error over the batch, (nE, B) -> (nE,).
        delta_theta = error.mean(dim=-1)
        new_theta = torch.clamp(
            params.theta + _per_run(eta, delta_theta.dim()) * delta_theta, min=0.0
        )
        return params.with_(theta=new_theta)

    raise ValueError(f"unknown model_code: {params.model_code!r}")


def eta_schedule(eta_0: float, epoch: int, decay: float, decay_start_epoch: int) -> float:
    """Return ``eta_0`` through ``decay_start_epoch`` (0-indexed).

    Each subsequent epoch multiplies the rate by another factor of ``decay``.
    """
    return eta_0 * decay ** max(0, epoch - decay_start_epoch)
