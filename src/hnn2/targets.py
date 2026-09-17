"""Compare target rates using saved per-seed measurements."""

TIEBREAK_ATOL = 1e-6
TIEBREAK_RULE = f"absolute difference <= {TIEBREAK_ATOL:g} from the best seed median (rtol=0): lower targ wins"
TARGET_CRITERION = "median_final_val_loss"


def classifier_target_candidates(frame):
    """Select each model's targ by final classifier validation loss.

    Pass trained representations only, with one row per model/targ/seed.
    Reject missing/nonfinite losses or unequal seed sets instead of silently
    choosing a target from fewer observations. No training or file writes.
    """
    import numpy as np
    import pandas as pd

    keys = ["model_code", "targ", "seed_index"]
    if not set(keys + ["final_val_loss"]).issubset(frame.columns):
        raise ValueError("Target selection requires saved final_val_loss and model/targ/seed")
    if frame.empty or frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError("Target selection needs unique nonempty model/targ/seed rows")
    losses = frame.final_val_loss.to_numpy(float)
    if not np.isfinite(losses).all() or (losses < 0).any():
        raise ValueError("Target selection needs finite nonnegative final_val_loss")

    seeds = frame.groupby(["model_code", "targ"]).seed_index.apply(lambda s: tuple(sorted(s)))
    if len(set(seeds)) != 1:
        raise ValueError("Target selection requires identical seed sets")

    candidates = _argmax_table(frame, "final_val_loss", maximize=False)
    return pd.DataFrame([
        {"model_code": model, "candidate_targ": value["chosen_targ"],
         "median_final_val_loss": value["value"], "boundary": value["boundary"],
         "n_seeds": len(seeds.iloc[0]), "criterion": TARGET_CRITERION,
         "tiebreak": TIEBREAK_RULE, "adoption": "not_performed"}
        for model, value in candidates.items()
    ])


def _argmax_table(df, value_col: str, *, maximize: bool = True) -> dict:
    """Best target per model variant under one criterion, plus the evidence table.

    ``df`` needs columns ``model_code``, ``targ`` and ``value_col``, one row per
    seed; seeds are collapsed to a median per target cell. Among medians within
    TIEBREAK_ATOL of the best value, the lowest targ wins (no relative tolerance).

    Returns per model: ``chosen_targ``, its ``value``, ``boundary`` (it sits at
    an end of the grid), the ``grid``, and ``cells`` -- every target's value.
    Variants with no usable data are omitted rather than given an entry.
    """
    import numpy as np

    out: dict[str, dict] = {}
    for model, sub in df.groupby("model_code"):
        cells = (sub.groupby("targ")[value_col].median().reset_index()
                 .sort_values(["targ"]))
        if cells.empty or cells[value_col].isna().all():
            continue

        best_value = cells[value_col].max() if maximize else cells[value_col].min()
        ties = cells[np.isclose(cells[value_col], best_value, rtol=0., atol=TIEBREAK_ATOL)]
        chosen_row = ties.iloc[0]  # cells are sorted by targ.
        chosen = float(chosen_row["targ"])
        grid = sorted(cells["targ"].tolist())
        out[model] = {
            "chosen_targ": chosen,
            "value": float(chosen_row[value_col]),
            "boundary": chosen in (grid[0], grid[-1]),
            "grid": grid,
            # NaN -> null: json.dumps would otherwise emit a non-standard NaN token.
            "cells": {str(float(t)): (None if np.isnan(v) else float(v))
                      for t, v in zip(cells["targ"], cells[value_col])},
        }
    return out
