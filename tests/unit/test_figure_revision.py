"""Classifier-loss selection and per-model figures, using fixed arrays only."""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from notebook_cells import SCORE_ARMS, SCORE_TABLES, draw_figures, table_results
from figure_fixtures import fixed_experiment
from hnn2.targets import classifier_target_candidates, TARGET_CRITERION
from hnn2.figure_tables import load_figure_runs
from hnn2.figure_plots import _histogram_axes


def loss_frame():
    # The better median deliberately has a much worse mean and composite.
    return pd.DataFrame([{"model_code": "rec", "targ": targ, "seed_index": seed,
        "final_val_loss": loss, "composite": 100 if targ == .55 else 0,
        "final_val_accuracy": 0 if targ == .55 else 1}
        for targ, losses in ((.35, (.5, .5, .5)), (.55, (.3, .4, 100)))
        for seed, loss in enumerate(losses)])


def test_classifier_uses_final_loss_median_not_mean_composite_or_accuracy():
    row = classifier_target_candidates(loss_frame()).iloc[0]
    assert (row.candidate_targ, row.median_final_val_loss, row.n_seeds) == (.55, .4, 3)
    assert row.criterion == TARGET_CRITERION


def test_classifier_loss_near_tie_keeps_chosen_actual_value():
    frame = loss_frame()
    frame.loc[frame.targ == .35, "final_val_loss"] = .4000005
    row = classifier_target_candidates(frame).iloc[0]
    assert row.candidate_targ == .35
    assert row.median_final_val_loss == .4000005


@pytest.mark.parametrize("damage", ["missing_seed", "duplicate", "nan", "inf", "negative", "missing_loss"])
def test_classifier_rejects_invalid_selection_evidence(damage):
    frame = loss_frame()
    if damage == "missing_seed": frame = frame.iloc[:-1]
    if damage == "duplicate": frame = pd.concat([frame, frame.iloc[:1]])
    if damage in ("nan", "inf", "negative"):
        frame.loc[0, "final_val_loss"] = {"nan": np.nan, "inf": np.inf, "negative": -.1}[damage]
    if damage == "missing_loss": frame = frame.drop(columns="final_val_loss")
    with pytest.raises(ValueError):
        classifier_target_candidates(frame)


@pytest.fixture(scope="module")
def fixed(tmp_path_factory):
    return fixed_experiment(tmp_path_factory.mktemp("figure_revision"))


def test_other_diagnostic_entry_uses_the_same_classifier_rule(fixed):
    from hnn2.postprocess import target_evidence
    paths = pd.read_csv(fixed / "plasticity/expected_runs.csv").path.tolist()
    frame, report = target_evidence(paths)
    assert report["selection_criterion"] == TARGET_CRITERION
    direct = classifier_target_candidates(frame)
    assert report["target_candidates"] == direct.to_dict("records")


@pytest.mark.parametrize("override", [None, {"rec": .35, "ff": .55, "thresh": .35}])
def test_model_specific_selection_reaches_all_figures_and_rf05(fixed, override):
    runs, expected = load_figure_runs(fixed)
    preferred = {"rec": .55, "ff": .35, "thresh": .55}
    for c, _, t in runs.values():
        if c["representation"] == "sparse_trained":
            s = c["spec"]
            t["summary"].loc[0, "final_val_loss"] = .1 if s["targ"] == preferred[s["model_code"]] else .4
    tables = table_results(runs, expected, fixed / "hp/selection", override)
    targets = override or preferred
    assert tables["analysis_targets"].set_index("model_code").targ.to_dict() == targets
    assert tables["target_candidate_evidence"].set_index("model_code").candidate_targ.to_dict() == preferred
    for name in ("selected_summary", "validation_points", "threshold_detail", "selected_entropy_scores"):
        frame = tables[name]
        trained = frame[frame.arm.str.endswith("_trained")]
        assert set(trained[["model_code", "targ"]].itertuples(index=False, name=None)) == set(targets.items())
    scores = tables["selected_entropy_scores"]
    assert len(scores) == 12  # Six arms, two paired seeds per arm.
    for name in SCORE_TABLES:
        assert set(tables[name].arm) == set(SCORE_ARMS), name
    assert scores.groupby('arm').seed_index.apply(set).to_dict() == dict.fromkeys(SCORE_ARMS, {0, 1})
    for name in ('selected_summary', 'silhouette_statistics', 'threshold_summary'):
        assert set(tables[name].arm) == set(SCORE_ARMS) | {'raw', 'mlp_frozen'}, name
    np.testing.assert_allclose(scores.S_overall, .5*(scores.S_pop+scores.S_dataset), equal_nan=True)
    assert tables["threshold_summary"].total.eq(2).all()
    figures = draw_figures(tables, runs, targets)
    try:
        params = {k:v for k,v in figures.items() if "parameter" in k}
        assert len(params) == 6
        for model, targ in targets.items():
            for seed in (0, 1):
                assert f"target={targ}" in params[f"G2_parameter_{model}_seed-{seed}"].axes[0].get_title()
        labels = figures["G5_validation_accuracy"].axes[0].get_legend_handles_labels()[1]
        assert all(f"{m}_trained (targ={t:g})" in labels for m,t in targets.items())
        assert [a.get_title() for a in figures["G3_S_violin"].axes] == ["S-pop", "S-dataset", "S-overall"]
        wanted_labels = [
            f'{arm} (targ={targets[arm.removesuffix("_trained")]:g})'
            if arm.endswith('_trained') else arm
            for arm in SCORE_ARMS
        ]
        for axis in figures['G3_S_violin'].axes:
            assert [label.get_text() for label in axis.get_xticklabels()] == wanted_labels
        for fig in figures.values():
            text = " ".join(a.get_title()+a.get_ylabel()+" ".join(a.get_legend_handles_labels()[1]) for a in fig.axes)
            assert "lifetime" not in text
    finally:
        for fig in figures.values(): plt.close(fig)


def test_frequency_break_preserves_x_and_every_bin_top():
    heights = np.array([.01, .02, .03, .8])
    fig = plt.figure()
    try:
        panels = _histogram_axes(fig, fig.add_gridspec(1, 1)[0], [heights])
        assert len(panels) == 2
        top, bottom = panels
        for ax in panels: ax.stairs(heights, np.arange(5))
        bottom.set_xlim(0, 4)
        assert top.get_xlim() == bottom.get_xlim() == (0, 4)
        omitted_lo, omitted_hi = bottom.get_ylim()[1], top.get_ylim()[0]
        assert omitted_lo < omitted_hi
        assert not ((heights > omitted_lo) & (heights < omitted_hi)).any()
        for ax in panels: np.testing.assert_array_equal(ax.patches[0].get_data().values, heights)
    finally:
        plt.close(fig)
