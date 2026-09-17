"""Run, resume, and select experiment conditions with per-run artifacts."""

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch

from .config import RunSpec, experiment_config_from_json
from .data import load_dataset
from .hp.engine import chunk, run_adaptation_batch
from .hp.rules import SelectionRule
from .hp.select import check_curves, select_grid, eta_table, landscape_table, best_configs_rows
from .result_io import (
    SaveOptions, _atomic, completed, finite_json, record_failure, save_single,
    write_result, single_files,
)
from .single import (
    AnalysisSettings, MlpSettings, _baseline_conditions, _validate_data,
    finish_adaptation, run_representation, scientific_conditions,
)


def condition_directory(root, spec):
    """Return the model/target/eta/seed path without creating directories."""
    return (
        Path(root) / spec.model_code
        / f"target-{float(spec.targ)}_eta-{float(spec.eta)}" / f"seed-{spec.seed_index}"
    )


def save_expected_runs(directory, rows, *, config=None):
    """Persist the intended population before work, including runs never started.

    This concrete table supplies RF-05's expected denominator after interruption.
    Existing populations cannot silently acquire different coordinates.
    """
    frame = pd.DataFrame(
        rows, columns=["representation", "model_code", "targ", "eta", "seed_index", "path"]
    )
    if config is not None:
        for key in ("readout_epochs", "readout_eval_every", "batch_size"):
            frame[key] = getattr(config, key)
    if frame.empty:
        raise ValueError("Expected runs must be nonempty and unique")
    frame["path"] = frame.path.map(lambda p: str(Path(p).resolve()))
    if frame.path.duplicated().any():
        raise ValueError("Expected runs must be nonempty and unique")
    frame = frame.sort_values("path", ignore_index=True)

    path = Path(directory) / "expected_runs.csv"
    if path.exists():
        previous = pd.read_csv(path, float_precision="round_trip").fillna("")
        if previous.path.duplicated().any():
            raise ValueError(f"Duplicate saved expected runs: {path}")
        existing = {r["path"]: r for r in previous.to_dict("records")}
        if any(existing.get(r["path"]) != r for r in frame.fillna("").to_dict("records")):
            raise ValueError(f"Expected run population differs: {path}; choose a new output directory")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic(path, lambda tmp: frame.to_csv(tmp, index=False))
    return frame


def baseline_population(directory, models, seeds, representations=("raw", "mlp_frozen")):
    """Return expected-run rows for initial models and shared raw/MLP baselines.

    Initial sparse models vary by model and seed; raw/MLP vary only by seed.
    Neither has a target or eta axis.
    """
    cases = [(f"{m}_initial", m) for m in models] + [(r, "") for r in representations]
    return [
        {"representation": name, "model_code": model, "targ": None, "eta": None,
         "seed_index": seed, "path": str(Path(directory) / name / f"seed-{seed}")}
        for seed in seeds for name, model in cases
    ]


def _groups(specs, batch_runs):
    specs = list(specs)
    if isinstance(batch_runs, bool) or not isinstance(batch_runs, int) or batch_runs < 1:
        raise ValueError("batch_runs must be a positive integer")
    if len(set(specs)) != len(specs) or not specs:
        raise ValueError("conditions must be nonempty and unique")
    for model in dict.fromkeys(spec.model_code for spec in specs):
        yield from chunk([spec for spec in specs if spec.model_code == model], batch_runs)


def _tensors(dataset, device):
    # Saved (samples, input units) arrays become (input units, samples) tensors.
    return {
        split: torch.as_tensor(
            dataset.arrays[f"x_{split}_small"].T, dtype=torch.float32, device=device
        )
        for split in ("train", "val", "test")
    }


def _check_saving(config, analysis, saving):
    if saving.save_full_test and not (
        config.readout_final_eval_full_test or analysis.split == "test_full"
    ):
        raise ValueError("Full-test features will not be computed; set save_full_test=False or enable computation explicitly")


def run_experiments(
    dataset_path, specs, config, directory, *, batch_runs=30, device="cpu",
    analysis=AnalysisSettings(), saving=SaveOptions(), script_path=None,
    settings_path=None, selection=None,
):
    """Skip complete conditions, batch only unfinished compatible runs, save per run.

    An unfinished condition is restarted from its beginning. A batch failure is
    reported for every affected condition; retrying with width 1 can isolate it.
    Return a table of complete, skipped, or failed states and failure reasons.
    """
    dataset = load_dataset(dataset_path)
    groups = list(_groups(specs, batch_runs))
    _check_saving(config, analysis, saving)
    if selection is not None and any(
        getattr(config, key) != value for key, value in selection["schedule"].items()
    ):
        raise ValueError("Selected schedule differs from config; use selected_condition's effective config")
    if selection is not None and selection["input"]["sha256"] != dataset.source["sha256"]:
        raise ValueError("Selection and experiment use different input datasets")

    # Record the population and check all saved conditions before adaptation starts.
    save_expected_runs(
        directory,
        [{"representation": "sparse_trained", **asdict(s),
          "path": str(condition_directory(directory, s))}
         for group in groups for s in group],
        config=config,
    )
    states, prepared = [], []
    for group in groups:
        pending = []
        for spec in group:
            _validate_data(
                dataset.arrays, config, spec,
                config.readout_final_eval_full_test or analysis.split == "test_full",
            )
            path = condition_directory(directory, spec)
            conditions = {
                **scientific_conditions(dataset, spec, config, analysis, device), **asdict(saving)
            }
            if selection is not None:
                evidence = _eta_evidence(selection, spec.model_code, spec.targ, eta=spec.eta)
                conditions["selection"] = {"rule": selection["rule"], **evidence}
                if "source" in selection:
                    conditions["selection"]["source"] = selection["source"]

            array_names, table_names = single_files(conditions)
            required = [f"{name}.npz" for name in array_names] + [f"{name}.csv" for name in table_names]
            if completed(path, conditions, required):
                states.append({**asdict(spec), "path": str(path), "state": "skipped", "reason": ""})
            else:
                pending.append((spec, path, conditions))
        prepared.append(pending)

    tensors = _tensors(dataset, device) if any(prepared) else None
    for pending in prepared:
        if not pending:
            continue
        try:
            with torch.inference_mode():
                adapted = run_adaptation_batch(
                    [item[0] for item in pending], config, tensors, torch.device(device)
                )
        except Exception as exc:
            for spec, path, conditions in pending:
                record_failure(path, conditions, exc)
                states.append({**asdict(spec), "path": str(path), "state": "failed", "reason": f"batch: {exc}"})
            continue

        # Readout, analysis, and saving remain independent for each adapted run.
        for (spec, path, conditions), result in zip(pending, adapted):
            try:
                output = finish_adaptation(
                    dataset, result, spec, config, analysis=analysis, device=device
                )
                output.conditions.update(engine="batched", batch_runs=len(pending))
                if "selection" in conditions:
                    output.conditions["selection"] = conditions["selection"]
                save_single(
                    output, path, script_path=script_path, settings_path=settings_path,
                    saving=saving, resume=path.exists(),
                )
                states.append({**asdict(spec), "path": str(path), "state": "complete", "reason": ""})
            except Exception as exc:
                record_failure(path, conditions, exc)
                states.append({**asdict(spec), "path": str(path), "state": "failed", "reason": str(exc)})
    return pd.DataFrame(states)


def run_baselines(
    dataset_path, models, seeds, config, directory, *,
    representations=("raw", "mlp_frozen"), mlp=MlpSettings(),
    analysis=AnalysisSettings(), device="cpu", saving=SaveOptions(),
    script_path=None, settings_path=None,
):
    """Run initial sparse, raw, and frozen-MLP baselines with linear classifiers.

    Reuse complete matching conditions; restart incomplete ones with their seeds.
    Save and return the status table, raising after saving if any baseline failed.
    """
    models, seeds, representations = list(models), list(seeds), list(representations)
    if not models or len(set(models)) != len(models) or not set(models) <= {"rec", "ff", "thresh"}:
        raise ValueError("Choose unique baseline models from rec, ff, thresh")
    if (
        not seeds or len(set(seeds)) != len(seeds)
        or any(type(seed) is not int or seed < 0 for seed in seeds)
    ):
        raise ValueError("Choose unique nonnegative integer baseline seeds")
    if (
        len(set(representations)) != len(representations)
        or not set(representations) <= {"raw", "mlp_frozen"}
    ):
        raise ValueError("Choose unique additional baselines from raw, mlp_frozen")
    _check_saving(config, analysis, saving)
    if "mlp_frozen" in representations:
        mlp.config(config)  # Validate its separate budget before any baseline runs.

    # Check the full baseline population before starting any pending run.
    dataset = load_dataset(dataset_path)
    save_expected_runs(
        directory, baseline_population(directory, models, seeds, representations), config=config
    )
    cases = [("sparse_initial", model) for model in models] + [(name, "rec") for name in representations]
    pending, states = [], []
    for seed in seeds:
        for representation, model in cases:
            name = f"{model}_initial" if representation == "sparse_initial" else representation
            path = Path(directory) / name / f"seed-{seed}"
            spec = RunSpec(model, 1., 1., seed)  # Only model and seed apply to baselines.
            _validate_data(
                dataset.arrays, config, spec,
                config.readout_final_eval_full_test or analysis.split == "test_full",
                size="big" if representation == "raw" else "small",
            )
            conditions = {
                **_baseline_conditions(dataset, spec, config, analysis, device, representation, mlp),
                **asdict(saving),
            }
            arrays, tables = single_files(conditions)
            required = [f"{n}.npz" for n in arrays] + [f"{n}.csv" for n in tables]
            row = {"representation": name, "seed_index": seed, "path": str(path)}
            if completed(path, conditions, required):
                states.append({**row, "state": "skipped", "reason": ""})
            else:
                pending.append((representation, model, path, conditions, row))

    for representation, model, path, conditions, row in pending:
        try:
            result = run_representation(
                dataset, config, row["seed_index"], representation=representation,
                model_code=model, mlp=mlp, analysis=analysis, device=device,
            )
            save_single(
                result, path, saving=saving, script_path=script_path,
                settings_path=settings_path, resume=path.exists(),
            )
            states.append({**row, "state": "complete", "reason": ""})
        except Exception as exc:
            record_failure(path, conditions, exc)
            states.append({**row, "state": "failed", "reason": str(exc)})

    states = pd.DataFrame(states)
    states.to_csv(Path(directory) / "last_run.csv", index=False)
    if states.state.eq("failed").any():
        raise RuntimeError(f"Some baselines failed; see {Path(directory) / 'last_run.csv'}")
    return states


def run_sweep(
    dataset_path, specs, config, directory, *, batch_runs=30, device="cpu",
    script_path=None, settings_path=None,
):
    """Save candidate monitor curves and return all candidate/epoch rows.

    Reuse completed candidates and restart unfinished ones. The returned table
    carries input provenance in its attributes; any failed candidate raises.
    """
    dataset = load_dataset(dataset_path)
    specs = list(specs)
    groups = list(_groups(specs, batch_runs))
    tensors = _tensors(dataset, device)
    arm = f"d{config.eta_decay:g}_s{config.eta_decay_start_epoch}"
    frames = []
    failures = []

    for group in groups:
        pending = []
        for spec in group:
            _validate_data(dataset.arrays, config, spec, False)
            path = condition_directory(directory, spec)
            conditions = {
                "spec": asdict(spec), "config": asdict(config), "input": dataset.source,
                "representation": "sweep", "device": str(torch.device(device)),
            }
            if completed(path, conditions, ["curves.csv"]):
                frames.append(pd.read_csv(path / "curves.csv", float_precision="round_trip"))
            else:
                pending.append((spec, path, conditions))

        if not pending:
            continue
        try:
            with torch.inference_mode():
                results = run_adaptation_batch(
                    [item[0] for item in pending], config, tensors, torch.device(device),
                    keep_trajectory=False, analysis_probes=False,
                )
            for (spec, path, conditions), result in zip(pending, results):
                conditions.update(engine="batched", batch_runs=len(pending))
                # Keep candidate eta distinct from its decayed value at each epoch.
                frame = pd.DataFrame([
                    {"arm": arm, **asdict(spec), **{k: v for k, v in row.items() if k != "eta"},
                     "eta_epoch": row["eta"]}
                    for row in result.monitor_records
                ])
                write_result(
                    path, conditions, tables={"curves": frame}, script_path=script_path,
                    settings_path=settings_path, resume=path.exists(),
                )
                frames.append(frame)
        except Exception as exc:
            for spec, path, conditions in pending:
                if not (path / "COMPLETE").exists():
                    record_failure(path, conditions, exc)
                    failures.append({**asdict(spec), "reason": str(exc)})

    # Require complete candidate/epoch coverage before exposing curves to selection.
    if failures:
        raise RuntimeError(f"Sweep has failed conditions; rerun to retry them: {failures}")
    curves = pd.concat(frames, ignore_index=True)
    check_curves(curves)
    expected = {
        (s.model_code, s.targ, s.eta, s.seed_index, epoch)
        for s in specs for epoch in range(config.n_epochs)
    }
    actual = set(
        curves[["model_code", "targ", "eta", "seed_index", "epoch"]]
        .itertuples(index=False, name=None)
    )
    if expected != actual:
        raise ValueError("Sweep curves contain missing or extra candidate/epoch rows")
    curves = curves.sort_values(
        ["model_code", "targ", "eta", "seed_index", "epoch"], ignore_index=True
    )
    curves.attrs["input"] = dataset.source
    return curves


def _selection_conditions(specs, config, rule, input_source, training_epochs):
    if (
        isinstance(training_epochs, bool) or not isinstance(training_epochs, int)
        or not rule.at_epoch < training_epochs <= config.n_epochs
    ):
        raise ValueError("Require at_epoch < training_epochs <= sweep n_epochs")
    if rule.at_epoch + rule.stability_epochs >= config.n_epochs:
        raise ValueError("Sweep is too short for the stability window")
    if not input_source or not input_source.get("sha256"):
        raise ValueError("Saved curves need their recorded input_source")
    return {
        "representation": "hp_selection", "config": asdict(config), "input": input_source,
        "sweep_specs": [asdict(s) for s in specs], "selection": asdict(rule),
        "training_epochs": training_epochs,
    }


def _selection_report(curves, specs, config, rule, input_source, training_epochs):
    """Build the same report for saving and for checking saved numerical evidence."""
    check_curves(curves)
    arm = f"d{config.eta_decay:g}_s{config.eta_decay_start_epoch}"
    expected = {
        (arm, s.model_code, s.targ, s.eta, s.seed_index, e)
        for s in specs for e in range(config.n_epochs)
    }
    actual = set(
        curves[["arm", "model_code", "targ", "eta", "seed_index", "epoch"]]
        .itertuples(index=False, name=None)
    )
    if actual != expected:
        raise ValueError("HP curve coverage differs from all candidate/epoch coordinates")

    selection = select_grid(curves, specs, config, rule)
    if len(selection["cells"]) != len({(s.model_code, s.targ) for s in specs}):
        raise ValueError("HP target keys collide at the recorded precision")
    selection.update(
        input=input_source, sweep_config=asdict(config),
        schedule={
            "eta_decay": config.eta_decay, "eta_decay_start_epoch": config.eta_decay_start_epoch,
            "n_epochs": training_epochs,
        },
    )
    return selection


def select_sweep(
    curves, specs, config, rule, directory, *, training_epochs,
    script_path=None, settings_path=None, input_source=None, resume=False,
):
    """Apply the selection rule to curves and save and return the selection report.

    Use explicit input provenance or the curves' input attribute. No model runs;
    training_epochs records the schedule for subsequent selected experiments.
    """
    specs = list(specs)
    input_source = input_source or curves.attrs.get("input")
    conditions = _selection_conditions(specs, config, rule, input_source, training_epochs)
    selection = _selection_report(curves, specs, config, rule, input_source, training_epochs)
    write_result(
        directory, conditions,
        tables={
            "curves": curves, "selected": eta_table(selection),
            "landscape": landscape_table(selection),
        },
        documents={"selection": selection}, script_path=script_path,
        settings_path=settings_path, resume=resume,
    )
    return selection


def read_selection(directory):
    """Read a completed selection and check it against its saved candidate curves.

    Only numerical tables are evaluated here; no dataset or network is loaded.
    Return the selection with source provenance and its saved sweep RunSpecs,
    whose model/target coordinates and seeds define the training population.
    """
    path = Path(directory).resolve()
    if not (path / "COMPLETE").is_file():
        raise ValueError(f"Incomplete HP selection: {path}")

    saved = json.loads((path / "conditions.json").read_text(encoding="utf-8"))
    if "training_epochs" not in saved:
        raise ValueError("HP selection lacks training_epochs; reselect the saved curves explicitly")
    specs = [RunSpec(**row) for row in saved["sweep_specs"]]
    config = experiment_config_from_json(json.dumps(saved["config"]))
    rule = SelectionRule(**saved["selection"])
    expected = _selection_conditions(specs, config, rule, saved["input"], saved["training_epochs"])
    completed(path, expected, ["selection.json", "curves.csv", "selected.csv", "landscape.csv"])

    # Recompute numerical evidence before trusting the saved selection and tables.
    curves = pd.read_csv(path / "curves.csv", float_precision="round_trip")
    recomputed = _selection_report(
        curves, specs, config, rule, saved["input"], saved["training_epochs"]
    )
    content = (path / "selection.json").read_bytes()
    selection = json.loads(content)
    if selection != finite_json(recomputed):
        raise ValueError("Saved selection differs from its candidate curves or conditions")
    for name, expected_table in (
        ("selected", eta_table(recomputed)), ("landscape", landscape_table(recomputed))
    ):
        saved_table = pd.read_csv(path / f"{name}.csv", float_precision="round_trip")
        if (
            list(saved_table.columns) != list(expected_table.columns)
            or saved_table.fillna("").to_dict("records")
            != expected_table.fillna("").to_dict("records")
        ):
            raise ValueError(f"Saved {name} table differs from HP selection")

    selection["source"] = {"path": str(path), "selection_sha256": hashlib.sha256(content).hexdigest()}
    return selection, specs


def _check_training_config(sweep_config, config):
    # Classifier settings may differ. All shared model/adaptation conditions must agree.
    differences = [key for key, value in asdict(sweep_config).items()
                   if key != "n_epochs" and not key.startswith("readout_")
                   and value != getattr(config, key)]
    if differences:
        raise ValueError(f"HP and training conditions differ: {differences}")


def run_sweep_selection(
    dataset_path, specs, config, rule, directory, selection_directory, *,
    training_config, batch_runs=30, device="cpu", script_path=None, settings_path=None,
):
    """Return a matching saved selection, or run and save its candidate sweep.

    Shared adaptation settings must match training_config; its epoch count sets
    the selected training schedule, while classifier settings may differ.
    """
    _check_training_config(config, training_config)
    dataset = load_dataset(dataset_path)
    specs = list(specs)
    conditions = _selection_conditions(specs, config, rule, dataset.source, training_config.n_epochs)
    if completed(
        selection_directory, conditions,
        ["selection.json", "curves.csv", "selected.csv", "landscape.csv"],
    ):
        return read_selection(selection_directory)[0]

    curves = run_sweep(
        dataset, specs, config, directory, batch_runs=batch_runs, device=device,
        script_path=script_path, settings_path=settings_path,
    )
    return select_sweep(
        curves, specs, config, rule, selection_directory,
        training_epochs=training_config.n_epochs, script_path=script_path,
        settings_path=settings_path, resume=Path(selection_directory).exists(),
    )


def run_selected_experiments(
    selection_directory, dataset_path, config, directory, *, batch_runs=30,
    device="cpu", analysis=AnalysisSettings(), saving=SaveOptions(),
    script_path=None, settings_path=None,
):
    """Train every saved model/target at its selected eta, using the HP seed set.

    Reuse complete conditions. Save and return the status table, raising after
    saving if any condition failed.
    """
    selection, sweep_specs = read_selection(selection_directory)
    sweep_config = experiment_config_from_json(json.dumps(selection["sweep_config"]))
    _check_training_config(sweep_config, config)
    if config.n_epochs != selection["schedule"]["n_epochs"]:
        raise ValueError("Training n_epochs differs from the recorded selection schedule")

    rows = best_configs_rows(selection)
    seed_sets = [
        {s.seed_index for s in sweep_specs
         if (s.model_code, s.targ) == (row["model_code"], row["targ"])}
        for row in rows
    ]
    if any(seeds != seed_sets[0] for seeds in seed_sets):
        raise ValueError("All HP model/target cells must use the same seed set")
    specs = [RunSpec(**row, seed_index=seed) for row in rows
             for seed in sorted(seed_sets[0])]

    states = run_experiments(
        dataset_path, specs, config, directory, selection=selection,
        batch_runs=batch_runs, device=device, analysis=analysis,
        saving=saving, script_path=script_path, settings_path=settings_path,
    )
    states.to_csv(Path(directory) / "last_run.csv", index=False)
    if states.state.eq("failed").any():
        raise RuntimeError(f"Some conditions failed; see {Path(directory) / 'last_run.csv'}")
    return states


def _eta_evidence(selection, model, targ, *, eta=None):
    cells = [c for c in selection["cells"].values() if c["model_code"] == model and c["targ"] == targ]
    if len(cells) != 1:
        raise ValueError("Selection needs exactly one matching model/target")
    chosen = cells[0]["selected_eta"]
    return {"selected_eta": chosen, "used_eta": chosen if eta is None else eta}


def selected_condition(selection, model, targ, seed, config, *, eta=None):
    """Return a RunSpec, config with the selected schedule, and eta evidence.

    An explicit eta overrides the value used, while preserving the selected eta
    in the evidence. The supplied config is not mutated.
    """
    evidence = _eta_evidence(selection, model, targ, eta=eta)
    effective = replace(config, **selection["schedule"])
    return RunSpec(model, targ, evidence["used_eta"], seed), effective, evidence
