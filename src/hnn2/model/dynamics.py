"""One settling-loop update as a pure function (C2; D1, D2).

Discrete map, per variant (dec = exp(-1/tau)):

  rec:    v_E(t) = dec_E v_E + W_EIn x - W_EI v_I ;  v_I(t) = dec_I v_I + W_IE o_E(t-1)
  ff:     v_E(t) = dec_E v_E + W_EIn x - W_EI v_I ;  v_I(t) = dec_I v_I + W_IIn x
  thresh: v_E(t) = dec_E v_E + W_EIn x

  o_E(t) = ReLU(v_E(t) - theta)     for all variants

Every right-hand side reads ``state`` at t-1, so v_E and v_I advance
simultaneously; only ``o_E`` uses the freshly updated v_E. Steady-state gains:
SPEC.md section 1 (I-09).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .params import ModelParams, State


@dataclass(frozen=True)
class Decays:
    """Per-tick leak factors: the fraction of potential surviving one update."""

    dec_E: float
    dec_I: float

    @classmethod
    def from_taus(cls, tau_e: float, tau_i: float) -> "Decays":
        # Exact discretisation: a fraction exp(-1/tau) survives one tick.
        return cls(dec_E=math.exp(-1.0 / tau_e), dec_I=math.exp(-1.0 / tau_i))


def input_drive(params: ModelParams, x: torch.Tensor) -> torch.Tensor:
    """Return the constant excitatory drive W_EIn @ x for ``x`` of shape (nIn, B).

    The result is (nE, B), or (R, nE, B) for stacked networks sharing ``x``.
    """
    return params.w_EIn @ x


def step(
    params: ModelParams,
    state: State,
    drive: torch.Tensor,
    decays: Decays,
    x: torch.Tensor,
) -> State:
    """Advance the settling map by one update; ``drive`` must be ``input_drive(params, x)``.

    Returns a new ``State`` without writing into its arguments. Shapes:
    ``drive``, ``v_E`` and ``o_E`` are (nE, B); ``x`` is (nIn, B); ``v_I`` is
    (nI, B), or (0, B) for thresh. Stacked params (SPEC V12) prepend a run
    axis R to every weight and state tensor; ``torch.matmul`` broadcasts it,
    and a 2-D ``x`` is then shared by all R networks.
    """
    # (nE,) -> (nE, 1) so each unit's threshold broadcasts across the batch;
    # unsqueeze(-1) does the same for stacked (R, nE) thresholds.
    theta = params.theta.unsqueeze(-1)

    if params.model_code == "rec":
        # E drives I, closing the negative-feedback loop.
        v_E = state.v_E * decays.dec_E + drive - params.w_EI @ state.v_I
        v_I = state.v_I * decays.dec_I + params.w_IE @ state.o_E
    elif params.model_code == "ff":
        # I integrates the stimulus directly, independently of E's response.
        v_E = state.v_E * decays.dec_E + drive - params.w_EI @ state.v_I
        v_I = state.v_I * decays.dec_I + params.w_IIn @ x
    elif params.model_code == "thresh":
        # Control condition: no inhibitory current; only theta can rein activity in.
        v_E = state.v_E * decays.dec_E + drive
        # Empty (0, B) tensor, carried through to keep State uniform across variants.
        v_I = state.v_I
    else:
        raise ValueError(f"unknown model_code: {params.model_code!r}")

    o_E = torch.clamp(v_E - theta, min=0.0)
    return State(v_E=v_E, o_E=o_E, v_I=v_I)


def analytic_fixed_point(
    params: ModelParams,
    x: torch.Tensor,
    decays: Decays,
    o_E_settled: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Return steady-state potentials to verify settling in tests (D2, Q2).

    These equations do not replace the scientific settling loop.
    ``rec`` requires the settled ``o_E``: its fixed point is piecewise-linear,
    so the ReLU support is known only after settling. ``thresh`` and ``ff``
    need the input alone.
    """
    # Steady-state amplification of a constant input, one factor per population.
    gain_E = 1.0 / (1.0 - decays.dec_E)
    gain_I = 1.0 / (1.0 - decays.dec_I)
    drive = input_drive(params, x)

    if params.model_code == "thresh":
        return {"v_E": gain_E * drive}
    if params.model_code == "ff":
        # I sees a constant, so its limit is exact; E then integrates the net current.
        v_I = gain_I * (params.w_IIn @ x)
        return {"v_E": gain_E * (drive - params.w_EI @ v_I), "v_I": v_I}
    if params.model_code == "rec":
        if o_E_settled is None:
            raise ValueError("rec fixed point requires the settled o_E")
        # Treat the settled E output as the constant input to I, then reuse ff's limit.
        v_I = gain_I * (params.w_IE @ o_E_settled)
        return {"v_E": gain_E * (drive - params.w_EI @ v_I), "v_I": v_I}
    raise ValueError(f"unknown model_code: {params.model_code!r}")
