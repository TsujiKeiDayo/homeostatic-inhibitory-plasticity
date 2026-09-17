"""Fixed numerical fixtures for artifact tests; never call a scientific run.

Weights/features/probes come from checked-in synthetic references. Readout
histories and final losses below are hand-authored test values, not reconstructed
measurements of the old run. All output goes to pytest's temporary directories.
"""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hnn2.adapt import AdaptationResult
from hnn2.config import ExperimentConfig, RunSpec
from hnn2.data import load_dataset
from hnn2.model.params import ModelParams
from hnn2.readout import ReadoutHistory, ReadoutResult
from hnn2.result_io import SaveOptions, save_single
from hnn2.single import AnalysisSettings, MlpSettings, SingleResult, scientific_conditions, _baseline_conditions
from hnn2.workflows import condition_directory, save_expected_runs, baseline_population, select_sweep
from hnn2.hp.rules import SelectionRule

BASE = Path(__file__).parent / "fixtures/m3_reference"
FROZEN = json.loads((BASE / "conditions.json").read_text(encoding="utf-8"))
CONFIG = ExperimentConfig(**{k: tuple(v) if k in ("image_size", "raw_baseline_size") else v
                            for k, v in FROZEN["config"].items()})


def arrays(path):
    with np.load(path, allow_pickle=False) as stored:
        return {k: stored[k] for k in stored.files}


def fixed_result(model="rec", seed=0, targ=.35, representation="sparse_trained"):
    data = load_dataset(BASE / "dataset.npz")
    spec = RunSpec(model, targ, .001, seed)
    name = f"{model}_{'trained' if representation == 'sparse_trained' else 'initial'}"
    if representation in ("raw", "mlp_frozen"):
        name = "raw" if representation == "raw" else "mlp"
    source = BASE / f"{name}_s{seed}"
    features = arrays(source / "features.npz")
    n = len(features["h_train"])
    batches = (n + CONFIG.batch_size - 1) // CONFIG.batch_size
    steps = CONFIG.readout_epochs * batches
    h = ReadoutHistory()
    val_loss = val_acc = float("nan")
    for step in range(1, steps + 1):
        evaluated = step % CONFIG.readout_eval_every == 0
        if evaluated:
            val_loss, val_acc = 1 / (1 + step), .3 + .05 * step
        for key, value in {"step": step, "epoch": (step - 1) // batches,
                           "batch_loss": 1 / step, "batch_accuracy": .5,
                           "val_loss": val_loss, "val_accuracy": val_acc,
                           "l1_delta": .1, "cumulative_l1": .1 * step,
                           "val_evaluated": evaluated}.items():
            getattr(h, key).append(value)
    summary = ReadoutResult(h, .3 + .05 * steps, .3 + .05 * steps,
                            .65 + .05 * seed, .7 + .05 * seed,
                            features["h_train"].shape[1], 3,
                            final_val_loss=1 / (1 + steps), final_step=steps)
    adapted, weights = None, None
    if representation == "sparse_trained":
        initial = ModelParams.from_arrays(model, arrays(source / "weights_initial.npz"), torch.device("cpu"))
        trained = ModelParams.from_arrays(model, arrays(source / "weights_trained.npz"), torch.device("cpu"))
        q, a = arrays(source / "summaries.npz"), arrays(source / "activity.npz")
        adapted = AdaptationResult(initial, trained, pd.read_csv(source / "monitor.csv").to_dict("records"),
                                   list(q["param_trajectory"]),
                                   list(zip(q["trajectory_epoch"], q["trajectory_batch_start"])))
        for epoch in {0, CONFIG.n_epochs - 1}:
            prefix = f"probe_ep{epoch:02d}_"
            adapted.analysis_probes[epoch] = {k[len(prefix):]: v for k,v in q.items() if k.startswith(prefix)}
            adapted.analysis_activity[epoch] = a[f"o_E_ep{epoch:02d}"]
        c = scientific_conditions(data, spec, CONFIG, AnalysisSettings(), "cpu")
    else:
        c = _baseline_conditions(data, spec, CONFIG, AnalysisSettings(), "cpu", representation,
                                 MlpSettings(**FROZEN["mlp"]))
        if representation == "sparse_initial":
            weights = arrays(BASE / f"{model}_trained_s{seed}/weights_initial.npz")
        elif representation == "mlp_frozen":
            weights = arrays(source / "weights.npz")
    c["fixture_note"] = "hand-authored current-format I/O fixture; no new scientific experiment"
    return SingleResult(adapted, features, summary, pd.read_csv(source / "entropy.csv"),
                        pd.read_csv(source / "silhouette.csv"), c, weights)


def fixed_experiment(root):
    """3 models x 2 targs x 2 seeds, all baselines; save a fixed HP selection."""
    results = Path(root) / "results"
    curves = pd.read_csv(BASE / "curves.csv", float_precision="round_trip")
    hp_specs = [RunSpec(*row) for row in curves[["model_code", "targ", "eta", "seed_index"]].drop_duplicates().itertuples(index=False, name=None)]
    from dataclasses import replace
    hp_config = replace(CONFIG, n_epochs=int(curves.epoch.max()) + 1)
    selection = select_sweep(curves, hp_specs, hp_config, SelectionRule(**FROZEN["hp_rule"]), results / "hp/selection",
                            training_epochs=CONFIG.n_epochs, input_source=load_dataset(BASE / "dataset.npz").source)
    specs = [RunSpec(m, t, selection["cells"][f"{m}@{t:g}"]["selected_eta"], s)
             for m in ("rec", "ff", "thresh") for t in (.35, .55) for s in (0, 1)]
    saving = SaveOptions(save_full_test=True)
    rows = []
    for spec in specs:
        path = condition_directory(results / "plasticity", spec)
        result = fixed_result(spec.model_code, spec.seed_index, spec.targ)
        result.conditions["spec"]["eta"] = spec.eta
        save_single(result, path, saving=saving)
        rows.append({"representation": "sparse_trained", **asdict(spec), "path": str(path)})
    save_expected_runs(results / "plasticity", rows, config=CONFIG)
    rows = baseline_population(results / "baselines", ("rec", "ff", "thresh"), (0, 1))
    for row in rows:
        rep = "sparse_initial" if row["model_code"] else row["representation"]
        save_single(fixed_result(row["model_code"] or "rec", row["seed_index"], representation=rep),
                    row["path"], saving=saving)
    save_expected_runs(results / "baselines", rows, config=CONFIG)
    return results
