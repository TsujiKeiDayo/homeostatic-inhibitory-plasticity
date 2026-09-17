"""Parameters of the eta selection objective and forward stability filter."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass


@dataclass(frozen=True)
class SelectionRule:
    """Configure eta selection by argmin with a stability filter (SPEC V11).

    Every parameter is recorded in the selection report.

    at_epoch          zero-based epoch whose seed-mean score is minimised;
                      7 means after eight epochs. Production duration is
                      specified separately when saving the selection.
    var_weight        lambda in score = l_mean + lambda * rate_var
                      (1.0 = per-unit MSE, SPEC V10; the dissertation used 0.3)
    stability_epochs  N: the window at_epoch + 1 .. at_epoch + N, read from
                      the sweep's longer run (prefix property)
    stability_tol     an eta is a candidate only if its seed-mean score rises
                      by at most this fraction over that window,
                      max(score[window]) / score[at_epoch] - 1 <= tol; None
                      disables the filter (plain argmin)
    """

    at_epoch: int = 9
    var_weight: float = 1.0
    stability_epochs: int = 3
    stability_tol: float | None = 0.5

    def __post_init__(self) -> None:
        if self.at_epoch < 0:
            raise ValueError("at_epoch must be >= 0")
        if self.var_weight < 0:
            raise ValueError("var_weight must be >= 0")
        if self.stability_epochs < 1:
            raise ValueError("stability_epochs must be >= 1")
        if self.stability_tol is not None and self.stability_tol < 0:
            raise ValueError("stability_tol must be >= 0 or None")

    def to_json(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, raw: dict) -> "SelectionRule":
        return cls(**raw)
