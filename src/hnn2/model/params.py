"""Model parameters and settling-loop state as pure data (D1).

``w_AB`` holds weights *into* population A *from* population B, so ``w_EI``
is (nE, nI); ``In`` is the external input.

Tensors per variant (V2):

  rec:    w_EIn, w_EI, w_IE, theta
  ff:     w_EIn, w_EI, w_IIn, theta
  thresh: w_EIn, theta
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch


@dataclass
class ModelParams:
    """Weights of one network instance; tensors a variant does not use stay None.

    All weights are non-negative: inhibition is subtracted in ``dynamics.step``.
    """

    model_code: str
    w_EIn: torch.Tensor              # (nE, nIn)
    theta: torch.Tensor              # (nE,)
    w_EI: torch.Tensor | None = None  # (nE, nI)   rec/ff — the adaptive weight
    w_IE: torch.Tensor | None = None  # (nI, nE)   rec
    w_IIn: torch.Tensor | None = None  # (nI, nIn)  ff

    @property
    def n_excitatory(self) -> int:
        return int(self.w_EIn.shape[-2])

    @property
    def n_input(self) -> int:
        return int(self.w_EIn.shape[-1])

    @property
    def n_inhibitory(self) -> int:
        # 0 for thresh (no w_EI); that is what sizes its v_I as (0, B).
        if self.w_EI is not None:
            return int(self.w_EI.shape[-1])
        return 0

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Leading run dimensions shared by every tensor (SPEC V12).

        ``()`` for one network; ``(R,)`` for R networks stacked by ``stack``.
        """
        return tuple(self.w_EIn.shape[:-2])

    _TENSOR_FIELDS = ("w_EIn", "theta", "w_EI", "w_IE", "w_IIn")

    @classmethod
    def stack(cls, items: "list[ModelParams]") -> "ModelParams":
        """Stack single networks of one variant along a new leading run axis.

        Every tensor gains a first dimension of size ``len(items)``; tensors a
        variant does not use stay ``None``. Inverse of ``select``.
        """
        if not items:
            raise ValueError("stack needs at least one ModelParams")
        codes = {p.model_code for p in items}
        if len(codes) != 1:
            raise ValueError(f"cannot stack different variants: {sorted(codes)}")
        if any(p.batch_shape for p in items):
            raise ValueError("stack expects single networks (batch_shape == ())")

        stacked = {}
        for name in cls._TENSOR_FIELDS:
            tensors = [getattr(p, name) for p in items]
            if any(t is None for t in tensors):
                if not all(t is None for t in tensors):
                    raise ValueError(f"{name} is present in some networks but not all")
                stacked[name] = None
            else:
                stacked[name] = torch.stack(tensors, dim=0)
        return cls(model_code=items[0].model_code, **stacked)

    def select(self, index: int) -> "ModelParams":
        """Select one network; its tensors share storage with the stacked instance."""
        if not self.batch_shape:
            raise ValueError("select needs stacked params (batch_shape != ())")
        return ModelParams(
            model_code=self.model_code,
            **{
                name: (None if getattr(self, name) is None else getattr(self, name)[index])
                for name in self._TENSOR_FIELDS
            },
        )

    def clone(self) -> "ModelParams":
        """Copy all tensors so a parameter snapshot survives later updates."""
        return ModelParams(
            model_code=self.model_code,
            w_EIn=self.w_EIn.clone(),
            theta=self.theta.clone(),
            w_EI=None if self.w_EI is None else self.w_EI.clone(),
            w_IE=None if self.w_IE is None else self.w_IE.clone(),
            w_IIn=None if self.w_IIn is None else self.w_IIn.clone(),
        )

    def with_(self, **updates) -> "ModelParams":
        """Return a new container with replacements; unchanged tensors are shared."""
        return replace(self, **updates)

    def to_arrays(self) -> dict:
        """Export CPU NumPy arrays for NPZ output, omitting absent tensors.

        Tensors must not require gradients; CPU arrays may share their storage.
        """
        out = {"w_EIn": self.w_EIn.cpu().numpy(), "theta": self.theta.cpu().numpy()}
        for name in ("w_EI", "w_IE", "w_IIn"):
            tensor = getattr(self, name)
            if tensor is not None:
                out[name] = tensor.cpu().numpy()
        return out

    @classmethod
    def from_arrays(
        cls, model_code: str, arrays: dict, device: torch.device
    ) -> "ModelParams":
        """Load saved arrays as float32 tensors on ``device``; missing keys stay None."""
        def _get(name):
            if name not in arrays:
                return None
            return torch.as_tensor(arrays[name], dtype=torch.float32, device=device)

        return cls(
            model_code=model_code,
            w_EIn=_get("w_EIn"),
            theta=_get("theta"),
            w_EI=_get("w_EI"),
            w_IE=_get("w_IE"),
            w_IIn=_get("w_IIn"),
        )


@dataclass
class State:
    """Settling-loop state with tensors of shape (units, batch).

    Stacked networks prepend a run axis: (R, units, batch).

    Samples are COLUMNS, not rows: weight matrices left-multiply the state
    (``W @ v``), the transpose of the usual PyTorch (batch, features) layout.
    """

    v_E: torch.Tensor  # (nE, B) excitatory membrane
    o_E: torch.Tensor  # (nE, B) excitatory output (post-ReLU)
    v_I: torch.Tensor  # (nI, B) inhibitory membrane (zeros-shaped (0, B) for thresh)

    def clone(self) -> "State":
        return State(self.v_E.clone(), self.o_E.clone(), self.v_I.clone())


def zero_state(params: ModelParams, batch: int, device: torch.device) -> State:
    """Create a float32 zero state, preserving any leading run dimensions."""
    nE = params.n_excitatory
    nI = params.n_inhibitory
    # Stacked params (hp.engine) carry their leading run axis into the state.
    lead = params.batch_shape
    # nI == 0 for thresh, so v_I is an empty (0, B) tensor — no special cases.
    return State(
        v_E=torch.zeros((*lead, nE, batch), dtype=torch.float32, device=device),
        o_E=torch.zeros((*lead, nE, batch), dtype=torch.float32, device=device),
        v_I=torch.zeros((*lead, nI, batch), dtype=torch.float32, device=device),
    )
