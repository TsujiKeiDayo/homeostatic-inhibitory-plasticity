"""Selection rule v4 on synthetic curves whose outcome is known by construction."""

import numpy as np
import pandas as pd
import pytest

from hnn2.config import ExperimentConfig, RunSpec
from hnn2.hp.rules import SelectionRule
from hnn2.hp.select import (
    best_configs_rows,
    eta_table,
    landscape_table,
    select_eta,
    select_grid,
    sensitivity_table,
)

T = 16
ARM = "d0.95_s3"
RULE = SelectionRule(at_epoch=9, var_weight=1.0, stability_epochs=3, stability_tol=0.5)


def _curve(model, targ, eta, seed, j, rate=None):
    """One (model, targ, eta, seed) trace with per-epoch J (l = J/2, var = J/2)."""
    rows = []
    for epoch in range(T):
        r = targ if rate is None else rate[epoch]
        rows.append({"arm": ARM, "model_code": model, "targ": targ, "eta": eta,
                     "seed_index": seed, "epoch": epoch, "l_mean": 0.5 * j[epoch],
                     "rate_var": 0.5 * j[epoch], "rate_mean": r, "rate_wmean": r})
    return rows


def _cell(model="rec", targ=0.2, seeds=(0, 1)):
    """Four etas. A (1e-2): the lowest score at epoch 9, but a passing minimum —
    the score triples over epochs 10..12 -> unstable. B (3e-3): a plateau
    slightly above A's minimum -> the stable argmin. C (1e-3): flat, higher.
    D (3e-4): flat and huge (never converged)."""
    rows = []
    for seed in seeds:
        j_a = np.full(T, 5.0)
        j_a[7:13] = [1.5, 1.0, 0.5, 1.5, 1.5, 1.5]     # minimum at 9, climbs right after
        j_a[13:] = 0.5
        rows += _curve(model, targ, 1e-2, seed, j_a)
        rows += _curve(model, targ, 3e-3, seed, np.full(T, 0.8))
        rows += _curve(model, targ, 1e-3, seed, np.full(T, 1.2))
        rows += _curve(model, targ, 3e-4, seed, np.full(T, 50.0))
    return pd.DataFrame(rows)


def test_stability_filter_prefers_the_settled_eta_over_the_passing_minimum():
    result = select_eta(_cell(), 0.2, RULE)
    assert result["selected_eta"] == 3e-3 and result["reason"] == "argmin_stable"
    assert result["eta_unfiltered"] == 1e-2 and result["filter_changed"]
    assert result["score"] == pytest.approx(0.8)
    assert result["n_candidates"] == 3 and result["n_grid"] == 4
    table = {e["eta"]: e for e in result["table"]}
    assert not table[1e-2]["stable"] and table[1e-2]["fluctuation"] == pytest.approx(2.0)
    assert table[3e-3]["stable"] and table[3e-3]["fluctuation"] == pytest.approx(0.0)
    assert not result["boundary"]


def test_no_filter_is_the_plain_argmin():
    result = select_eta(_cell(), 0.2, SelectionRule(stability_tol=None))
    assert result["selected_eta"] == 1e-2 and result["reason"] == "argmin_stable"
    assert not result["filter_changed"]


def test_a_falling_score_counts_as_stable():
    cell = _cell()
    # B keeps improving after the horizon: negative rise, never disqualified.
    mask = (cell.eta == 3e-3) & (cell.epoch >= 10)
    cell.loc[mask, ["l_mean", "rate_var"]] = 0.2
    result = select_eta(cell, 0.2, RULE)
    table = {e["eta"]: e for e in result["table"]}
    assert table[3e-3]["fluctuation"] < 0 and table[3e-3]["stable"]
    assert result["selected_eta"] == 3e-3


def test_all_unstable_falls_back_to_the_argmin_and_says_so():
    cell = _cell()
    # Every eta's score doubles right after the horizon.
    mask = cell.epoch.isin([10, 11, 12])
    cell.loc[mask, ["l_mean", "rate_var"]] *= 2.0
    result = select_eta(cell, 0.2, RULE)
    assert result["reason"] == "argmin_unstable" and result["n_candidates"] == 0
    assert result["selected_eta"] == 1e-2


def test_tie_goes_to_the_smaller_eta():
    cell = _cell()
    cell.loc[cell.eta == 1e-3, ["l_mean", "rate_var"]] = 0.4      # J = 0.8, ties with 3e-3
    assert select_eta(cell, 0.2, RULE)["selected_eta"] == 1e-3


def test_boundary_flag_when_the_winner_sits_at_the_grid_edge():
    cell = _cell()
    cell = cell[cell.eta.isin([1e-2, 3e-3])]      # 3e-3 is now the grid bottom
    result = select_eta(cell, 0.2, RULE)
    assert result["selected_eta"] == 3e-3 and result["boundary"] and result["boundary_side"] == "low"


def test_short_curves_raise():
    with pytest.raises(ValueError, match="t_max"):
        select_eta(_cell(), 0.2, SelectionRule(at_epoch=T - 2, stability_epochs=3))


CONFIG = ExperimentConfig(n_epochs=T)

def _specs():
    return [RunSpec("rec", targ, eta, seed) for targ in (.2, .5)
            for eta in (3e-4, 1e-3, 3e-3, 1e-2) for seed in (0, 1)]


def _curves():
    return pd.concat([_cell("rec", 0.2), _cell("rec", 0.5)], ignore_index=True)


def test_select_grid_tables_every_cell():
    report = select_grid(_curves(), _specs(), CONFIG, RULE)
    assert report["n_cells"] == 2 and report["n_stable"] == 2 and report["n_unstable"] == 0
    assert report["n_filter_changed"] == 2 and report["rule"]["at_epoch"] == 9
    table = eta_table(report)
    assert table.eta.tolist() == [3e-3, 3e-3]
    assert table.stable.all() and (table.eta_unfiltered == 1e-2).all()
    assert best_configs_rows(report) == [{"model_code": "rec", "targ": 0.2, "eta": 3e-3},
                                         {"model_code": "rec", "targ": 0.5, "eta": 3e-3}]
    landscape = landscape_table(report)
    assert len(landscape) == 2 * 4
    assert landscape.selected.sum() == 2
    assert landscape[landscape.eta == 1e-2].stable.eq(False).all()


def test_select_grid_needs_the_plans_arm():
    curves = _curves()
    curves["arm"] = "d1_s0"
    with pytest.raises(ValueError, match="arm"):
        select_grid(curves, _specs(), CONFIG, RULE)


def test_sensitivity_table_flags_moved_cells():
    table = sensitivity_table(_curves(), _specs(), CONFIG, RULE,
                              {"stability_tol": [None, 0.5, 3.0], "at_epoch": [12]})
    off = table[(table.parameter == "stability_tol") & (table.value == "none")]
    assert off.changed.all() and set(off.eta) == {1e-2}
    loose = table[(table.parameter == "stability_tol") & (table.value == 3.0)]
    assert loose.changed.all()                    # a 200% rise passes a 300% tolerance
    same = table[(table.parameter == "stability_tol") & (table.value == 0.5)]
    assert not same.changed.any()
    # At epoch 12 eta A scores 1.5 and B (0.8) wins outright: no change either.
    late = table[table.parameter == "at_epoch"]
    assert not late.changed.any() and set(late.eta) == {3e-3}




def test_train_residual_is_reported_when_the_column_exists():
    curves = _curves()
    assert eta_table(select_grid(curves, _specs(), CONFIG, RULE)).residual_train.isna().all()
    curves["rate_wmean_train"] = curves.targ * 1.01          # 1 % off on the train split
    curves["rate_mean_train"] = curves.targ * 1.01
    table = eta_table(select_grid(curves, _specs(), CONFIG, RULE))
    assert np.allclose(table.residual_train, 0.01)
    assert "residual_train" in landscape_table(select_grid(curves, _specs(), CONFIG, RULE)).columns
