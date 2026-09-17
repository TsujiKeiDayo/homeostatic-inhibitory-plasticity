"""Reject damaged sweep evidence instead of selecting from silent pandas omissions."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from hnn2.config import ExperimentConfig, RunSpec
from hnn2.hp.rules import SelectionRule
from hnn2.hp.select import evaluate_eta, select_eta, select_grid


RULE = SelectionRule(at_epoch=1, stability_epochs=2)
CONFIG = ExperimentConfig(n_epochs=4)
SPECS = [RunSpec("rec", .2, eta, seed) for eta in (.001, .003) for seed in (0, 1)]


def curves():
    return pd.DataFrame([
        {"arm": "d0.95_s3", "model_code": "rec", "targ": 0.2, "eta": eta,
         "seed_index": seed, "epoch": epoch, "l_mean": score / 2,
         "rate_var": score / 2, "rate_mean": 0.2, "rate_wmean": 0.2}
        for eta, score in ((0.001, 0.2), (0.003, 0.1))
        for seed in range(2) for epoch in range(4)
    ])


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("column", ["l_mean", "rate_var", "rate_mean", "rate_wmean"])
def test_invalid_seed_is_not_silently_excluded_from_the_mean(column, value):
    table = curves()
    table.loc[(table.eta == 0.003) & (table.seed_index == 1) & (table.epoch == 1), column] = value
    with pytest.raises(ValueError, match=f"non-finite {column}"):
        select_grid(table, SPECS, CONFIG, RULE)


def test_nonfinite_stability_window_does_not_pass_as_zero_fluctuation():
    table = curves()
    table.loc[table.epoch == 3, "rate_var"] = np.nan
    with pytest.raises(ValueError, match="non-finite rate_var"):
        select_eta(table, 0.2, RULE)


def test_missing_one_seed_at_one_epoch_does_not_change_the_population():
    table = curves()
    table = table[~((table.eta == 0.003) & (table.seed_index == 1) & (table.epoch == 2))]
    with pytest.raises(ValueError, match="incomplete epoch/seed coverage"):
        select_eta(table, 0.2, RULE)


def test_duplicate_seed_row_is_not_counted_twice():
    table = curves()
    with pytest.raises(ValueError, match="duplicate epoch/seed"):
        select_eta(pd.concat([table, table.iloc[[1]]]), 0.2, RULE)


@pytest.mark.parametrize("drop", ["seed", "eta", "cell"])
def test_plan_requires_planned_runs(drop):
    table = curves()
    if drop == "seed":
        table = table[table.seed_index == 0]
    elif drop == "eta":
        table = table[table.eta == 0.001]
    else:
        specs = SPECS + [replace(s, targ=.5) for s in SPECS]
        with pytest.raises(ValueError, match="sweep run coverage.*missing"):
            select_grid(table, specs, CONFIG, RULE)
        return
    with pytest.raises(ValueError, match="sweep run coverage.*missing"):
        select_grid(table, SPECS, CONFIG, RULE)


def test_extra_seed_is_rejected_instead_of_replacing_a_planned_seed():
    table = curves().replace({"seed_index": {1: 7}})
    with pytest.raises(ValueError, match="missing.*extra"):
        select_grid(table, SPECS, CONFIG, RULE)


def test_candidates_use_the_same_seed_set_even_without_a_plan():
    table = curves()
    table = table[~((table.eta == 0.003) & (table.seed_index == 1))]
    with pytest.raises(ValueError, match="incomplete epoch/seed coverage"):
        select_eta(table, 0.2, RULE)


def test_score_std_and_seed_count_describe_the_actual_complete_population():
    table = curves().query("eta == 0.001").copy()
    table.loc[table.seed_index == 1, ["l_mean", "rate_var"]] = 0.2
    evaluated = evaluate_eta(table, 0.2, RULE)
    assert evaluated["n_seeds"] == 2
    assert evaluated["score"] == pytest.approx(0.3)
    assert evaluated["score_std"] == pytest.approx(0.1)


def test_zero_then_positive_is_unstable_but_zero_plateau_is_stable():
    table = curves()
    table.loc[(table.eta == 0.003) & (table.epoch == 1), ["l_mean", "rate_var"]] = 0.0
    result = select_eta(table, 0.2, RULE)
    rising = next(entry for entry in result["table"] if entry["eta"] == 0.003)
    assert np.isinf(rising["fluctuation"]) and not rising["stable"]
    assert result["selected_eta"] == 0.001
    table.loc[table.eta == 0.003, ["l_mean", "rate_var"]] = 0.0
    result = select_eta(table, 0.2, RULE)
    assert result["selected_eta"] == 0.003 and result["fluctuation"] == 0.0


def test_optional_legacy_diagnostic_can_be_absent_but_not_partly_missing():
    table = curves().query("eta == 0.001").copy()
    assert np.isnan(evaluate_eta(table, 0.2, RULE)["residual_train"])
    table["rate_wmean_train"] = np.nan
    assert np.isnan(evaluate_eta(table, 0.2, RULE)["residual_train"])
    table.loc[table.seed_index == 0, "rate_wmean_train"] = 0.2
    with pytest.raises(ValueError, match="rate_wmean_train"):
        evaluate_eta(table, 0.2, RULE)




@pytest.mark.parametrize("column,value", [("eta", np.nan), ("epoch", 1.5), ("seed_index", np.inf)])
def test_invalid_identity_is_rejected_before_groupby(column, value):
    table = curves().astype({column: float})
    table.loc[0, column] = value
    with pytest.raises(ValueError, match="identity|integer|non-finite"):
        select_eta(table, 0.2, RULE)
