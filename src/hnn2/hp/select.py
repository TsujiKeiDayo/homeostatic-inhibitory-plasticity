"""Select eta from the sweep's monitor curves using rule v4 (SPEC V11).

Input is the long-format curves table the sweep stage writes, one row per
(arm, model_code, targ, eta, seed_index, epoch) with the monitor statistics
``l_mean``, ``rate_var``, ``rate_mean``, ``rate_wmean``. Nothing here reads
a file or touches the GPU; ``workflows.select_sweep`` handles saving.

Per (model, targ) cell, for each eta (SPEC V11):

  1. its SCORE is the seed-mean composite l_mean + var_weight * rate_var at
     ``rule.at_epoch`` — the dissertation's §4.6.2 objective read at the last
     epoch of a T = at_epoch + 1 run (var_weight = 1 makes it the per-unit
     MSE, SPEC V10);
  2. its FLUCTUATION is how far the seed-mean score RISES over the
     ``rule.stability_epochs`` epochs AFTER ``at_epoch``:
     max(score[at+1 .. at+N]) / score[at] - 1 (negative when it keeps
     falling); the eta is STABLE when this is <= ``rule.stability_tol``.
     The sweep must cover the forward window after ``at_epoch``: an eta whose
     score at the horizon is a passing minimum — when its rate crosses the
     target — shows up as a score that climbs right afterwards. (A backward window
     on the RELATIVE score change cannot do this: near the fixed point the
     score is tiny and any settling looks like a large relative swing, so
     only etas that never learned pass; 2026-09-04 diagnosis.)
  3. the selected eta is the argmin of the score over the stable etas (ties:
     the smaller eta). A cell with no stable eta falls back to the unfiltered
     argmin and is flagged (``reason == "argmin_unstable"``).
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pandas as pd

from .rules import SelectionRule

CURVE_COLUMNS = ("arm", "model_code", "targ", "eta", "seed_index", "epoch",
                 "l_mean", "rate_var", "rate_mean", "rate_wmean")
CURVE_KEYS = CURVE_COLUMNS[:6]


def check_curves(curves: pd.DataFrame) -> None:
    """Check identity before groupby can silently drop missing keys or merge duplicates."""
    missing = [c for c in CURVE_COLUMNS if c not in curves.columns]
    if missing:
        raise ValueError(f"curves table lacks columns {missing}")
    if curves.empty:
        raise ValueError("curves table is empty")
    if curves[list(CURVE_KEYS)].isna().any().any():
        raise ValueError("curves have missing identity keys")
    duplicate = curves.duplicated(list(CURVE_KEYS), keep=False)
    if duplicate.any():
        sample = curves.loc[duplicate, list(CURVE_KEYS)].head(3).to_dict("records")
        raise ValueError(f"curves have duplicate epoch/seed rows: {sample}")

    for key in ("targ", "eta", "seed_index", "epoch"):
        values = curves[key].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"curves have non-finite {key}")
        if key in ("targ", "eta") and (values <= 0).any():
            raise ValueError(f"curves require positive {key}")
        if key in ("seed_index", "epoch") and ((values < 0) | (values != np.floor(values))).any():
            raise ValueError(f"curves require nonnegative integer {key}")


def _required_epochs(rule: SelectionRule) -> list[int]:
    if not np.isfinite(rule.var_weight) or rule.var_weight < 0:
        raise ValueError("var_weight must be finite and nonnegative")
    if rule.stability_tol is not None and (
            not np.isfinite(rule.stability_tol) or rule.stability_tol < 0):
        raise ValueError("stability_tol must be finite and nonnegative, or None")
    return list(range(rule.at_epoch, rule.at_epoch + rule.stability_epochs + 1))


def _required_rows(sub: pd.DataFrame, epochs: list[int], seeds) -> pd.DataFrame:
    """Require every seed at every scored epoch; a seed mean must not hide gaps."""
    absent = [epoch for epoch in epochs if epoch not in set(sub.epoch)]
    if absent:
        raise ValueError(
            f"curves lack epochs {absent}: the sweep's t_max must exceed "
            f"at_epoch + stability_epochs = {epochs[-1]}")
    expected = pd.MultiIndex.from_product([seeds, epochs], names=["seed_index", "epoch"])
    rows = sub[sub.epoch.isin(epochs)]
    actual = pd.MultiIndex.from_frame(rows[["seed_index", "epoch"]])
    missing, extra = expected.difference(actual), actual.difference(expected)
    if len(missing) or len(extra):
        raise ValueError(f"curves have incomplete epoch/seed coverage: "
                         f"missing={missing.tolist()[:5]}, extra={extra.tolist()[:5]}")
    for column in ("l_mean", "rate_var", "rate_mean", "rate_wmean"):
        values = rows[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"curves have non-finite {column} at required epoch/seed rows")
        if column in ("l_mean", "rate_var") and (values < 0).any():
            raise ValueError(f"curves require nonnegative {column}")
    return rows


def evaluate_eta(sub: pd.DataFrame, targ: float, rule: SelectionRule, *,
                 expected_seeds=None) -> dict:
    """Score, fluctuation and stability of one eta from its per-seed curves.

    ``sub`` holds every (seed_index, epoch) row of one (arm, model, targ, eta).
    """
    check_curves(sub)
    identities = sub[["arm", "model_code", "targ", "eta"]].drop_duplicates()
    if len(identities) != 1 or float(identities.targ.iloc[0]) != float(targ):
        raise ValueError("evaluate_eta needs exactly one (arm, model, targ, eta) matching targ")
    epochs = _required_epochs(rule)
    seeds = sorted(sub.seed_index.unique()) if expected_seeds is None else list(expected_seeds)
    required = _required_rows(sub, epochs, seeds)

    # Average the objective across matching seeds before testing its future rise.
    scored = required.assign(score=required.l_mean + rule.var_weight * required.rate_var)
    if not np.isfinite(scored.score.to_numpy(dtype=float)).all():
        raise ValueError("curves yield a non-finite composite score")
    per_epoch = scored.groupby("epoch").score.mean()          # seed mean
    window = epochs[1:]
    at_score = float(per_epoch.loc[rule.at_epoch])
    after = per_epoch.loc[window].to_numpy(dtype=float)
    if not np.isfinite(per_epoch.to_numpy(dtype=float)).all():
        raise ValueError("curves yield a non-finite seed-mean composite score")
    # A zero objective which subsequently rises has an unbounded relative
    # rise. Only a zero plateau has zero fluctuation.
    fluctuation = (
        float(after.max() / at_score - 1.0) if at_score > 0 else
        0.0 if after.max() == 0 else float("inf")
    )
    stable = rule.stability_tol is None or fluctuation <= rule.stability_tol

    at = scored[scored.epoch == rule.at_epoch]
    wmean = float(at.rate_wmean.mean())
    # Train-split fixed-point residual, descriptive only (absent in curves
    # written before 2026-09-05 -> NaN).
    wmean_train = float("nan")
    if "rate_wmean_train" in at.columns and not at.rate_wmean_train.isna().all():
        if not np.isfinite(at.rate_wmean_train.to_numpy(dtype=float)).all():
            raise ValueError("curves have incomplete or non-finite rate_wmean_train at the horizon")
        wmean_train = float(at.rate_wmean_train.mean())
    return {
        "eta": float(sub.eta.iloc[0]),
        "n_seeds": len(seeds),
        "score": float(per_epoch.loc[rule.at_epoch]),
        "score_std": float(at.score.std(ddof=0)),
        "fluctuation": fluctuation,
        "stable": bool(stable),
        "rate_mean": float(at.rate_mean.mean()),
        "rate_wmean": wmean,
        "residual": abs(wmean - targ) / targ,
        "residual_train": abs(wmean_train - targ) / targ,
    }


def _argmin(entries: list[dict]) -> dict:
    return min(entries, key=lambda entry: (entry["score"], entry["eta"]))


def select_eta(cell: pd.DataFrame, targ: float, rule: SelectionRule, *,
               expected_seeds=None) -> dict:
    """Rule v4 for one (model, targ) cell; ``cell`` holds all its etas.

    Returns the full audit record: the selected eta, the unfiltered argmin
    beside it, and every eta's score / fluctuation / stability.
    """
    check_curves(cell)
    identities = cell[["arm", "model_code", "targ"]].drop_duplicates()
    if len(identities) != 1 or float(identities.targ.iloc[0]) != float(targ):
        raise ValueError("select_eta needs exactly one (arm, model, targ) matching targ")
    seeds = sorted(cell.seed_index.unique()) if expected_seeds is None else list(expected_seeds)
    table = []
    for eta, sub in cell.groupby("eta", sort=True):
        try:
            table.append(evaluate_eta(sub, targ, rule, expected_seeds=seeds))
        except ValueError as exc:
            raise ValueError(f"eta={eta:g}: {exc}") from exc

    grid = [entry["eta"] for entry in table]
    stable = [entry for entry in table if entry["stable"]]
    chosen = _argmin(stable) if stable else _argmin(table)
    unfiltered = _argmin(table)
    return {
        "selected_eta": chosen["eta"],
        "score": chosen["score"],
        "fluctuation": chosen["fluctuation"],
        "rate_mean": chosen["rate_mean"],
        "rate_wmean": chosen["rate_wmean"],
        "residual": chosen["residual"],
        "residual_train": chosen["residual_train"],
        "n_candidates": len(stable),
        "n_grid": len(grid),
        "boundary": chosen["eta"] in (grid[0], grid[-1]),
        "boundary_side": ("low" if chosen["eta"] == grid[0]
                          else "high" if chosen["eta"] == grid[-1] else ""),
        "reason": "argmin_stable" if stable else "argmin_unstable",
        "eta_unfiltered": unfiltered["eta"],
        "filter_changed": not math.isclose(unfiltered["eta"], chosen["eta"]),
        "grid": grid,
        "table": table,
    }


def select_grid(curves: pd.DataFrame, specs, config, rule: SelectionRule) -> dict:
    """Rule v4 for an explicit list of RunSpecs under one schedule."""
    specs = list(specs)
    arm = f"d{config.eta_decay:g}_s{config.eta_decay_start_epoch}"
    check_curves(curves)
    sub = curves[curves.arm == arm]
    if sub.empty:
        raise ValueError(f"no curves for arm {arm!r}")
    epochs = _required_epochs(rule)
    if epochs[-1] >= config.n_epochs:
        raise ValueError(f"selection needs epoch {epochs[-1]}, beyond sweep t_max={config.n_epochs}")

    # Check the supplied run population before any per-cell selection.
    run_keys = ["model_code", "targ", "eta", "seed_index"]
    expected = pd.MultiIndex.from_tuples(
        [(s.model_code, s.targ, s.eta, s.seed_index) for s in specs], names=run_keys)
    if expected.has_duplicates or not len(expected):
        raise ValueError("sweep conditions must be nonempty and unique")
    actual = pd.MultiIndex.from_frame(sub[run_keys].drop_duplicates())
    missing, extra = expected.difference(actual), actual.difference(expected)
    if len(missing) or len(extra):
        raise ValueError(f"sweep run coverage differs: {len(missing)} missing, "
                         f"{len(extra)} extra; missing={missing.tolist()[:5]}, "
                         f"extra={extra.tolist()[:5]}")
    if (sub.epoch >= config.n_epochs).any():
        raise ValueError(f"curves contain epochs beyond sweep t_max={config.n_epochs}")

    cells: dict[str, dict] = {}
    for (model, targ), cell in sub.groupby(["model_code", "targ"], sort=True):
        try:
            result = select_eta(
                cell, float(targ), rule,
                expected_seeds=sorted({
                    s.seed_index for s in specs
                    if s.model_code == model and s.targ == targ
                })
            )
        except ValueError as exc:
            raise ValueError(f"{model}@{targ:g}: {exc}") from exc
        result.update(model_code=str(model), targ=float(targ))
        cells[f"{model}@{targ:g}"] = result
    values = list(cells.values())
    return {
        "rule": rule.to_json(),
        "arm": arm,
        "n_cells": len(cells),
        "n_stable": sum(c["reason"] == "argmin_stable" for c in values),
        "n_unstable": sum(c["reason"] == "argmin_unstable" for c in values),
        "n_boundary": sum(bool(c["boundary"]) for c in values),
        "n_filter_changed": sum(bool(c["filter_changed"]) for c in values),
        "cells": cells,
    }


def eta_table(report: dict) -> pd.DataFrame:
    """Flat per-cell view of ``select_grid``'s report (one row per cell)."""
    rows = []
    for cell in report["cells"].values():
        rows.append({
            "model_code": cell["model_code"],
            "targ": cell["targ"],
            "eta": cell["selected_eta"],
            "score": cell["score"],
            "fluctuation": cell["fluctuation"],
            "stable": cell["reason"] == "argmin_stable",
            "n_candidates": cell["n_candidates"],
            "boundary": cell["boundary"],
            "boundary_side": cell["boundary_side"],
            "reason": cell["reason"],
            "eta_unfiltered": cell["eta_unfiltered"],
            "filter_changed": cell["filter_changed"],
            "rate_mean": cell["rate_mean"],
            "rate_wmean": cell["rate_wmean"],
            "residual": cell["residual"],
            "residual_train": cell["residual_train"],
            "grid_min": cell["grid"][0],
            "grid_max": cell["grid"][-1],
        })
    return pd.DataFrame(rows).sort_values(["model_code", "targ"], ignore_index=True)


def landscape_table(report: dict) -> pd.DataFrame:
    """Every eta of every cell: the score landscape the argmin was read from."""
    rows = []
    for cell in report["cells"].values():
        for entry in cell["table"]:
            rows.append({
                "model_code": cell["model_code"],
                "targ": cell["targ"],
                "eta": entry["eta"],
                "score": entry["score"],
                "score_std": entry["score_std"],
                "fluctuation": entry["fluctuation"],
                "stable": entry["stable"],
                "selected": math.isclose(entry["eta"], cell["selected_eta"]),
                "rate_mean": entry["rate_mean"],
                "rate_wmean": entry["rate_wmean"],
                "residual": entry["residual"],
                "residual_train": entry["residual_train"],
            })
    return pd.DataFrame(rows).sort_values(["model_code", "targ", "eta"], ignore_index=True)


def best_configs_rows(report: dict) -> list[dict]:
    """The (model_code, targ, eta) rows production consumes."""
    return [
        {"model_code": cell["model_code"], "targ": cell["targ"], "eta": cell["selected_eta"]}
        for cell in report["cells"].values()
    ]


def sensitivity_table(curves: pd.DataFrame, specs, config, rule: SelectionRule,
                      variations: dict[str, list]) -> pd.DataFrame:
    """Vary one rule parameter at a time and flag changes from the selected eta."""
    specs = list(specs)
    base = eta_table(select_grid(curves, specs, config, rule)).set_index(["model_code", "targ"])
    rows = []
    for parameter, values in variations.items():
        for value in values:
            table = eta_table(
                select_grid(curves, specs, config, replace(rule, **{parameter: value}))
            )
            for key, row in table.set_index(["model_code", "targ"]).iterrows():
                base_eta = float(base.loc[key, "eta"])
                rows.append({
                    "parameter": parameter,
                    "value": "none" if value is None else value,
                    "model_code": key[0], "targ": key[1],
                    "eta": row["eta"], "eta_base": base_eta,
                    "changed": not math.isclose(float(row["eta"]), base_eta),
                    "reason": row["reason"],
                })
    return pd.DataFrame(rows)
