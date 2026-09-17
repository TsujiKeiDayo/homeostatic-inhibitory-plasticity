"""Explicit re-use of saved features and numeric tables; no upstream execution."""
from dataclasses import asdict
import hashlib
from io import BytesIO
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .result_io import readout_tables, write_result, recorded_options_issues
from .single import AnalysisSettings, analyse_features, fit_readout, validate_features
from .rng import stream


def _conditions(directory):
    path = Path(directory)
    if not (path / "COMPLETE").is_file():
        raise ValueError(f"Incomplete result: {path}")
    conditions = json.loads((path / "conditions.json").read_text(encoding="utf-8"))
    issues = recorded_options_issues(conditions)
    if issues:
        raise ValueError(f"Invalid saved conditions: {issues}")
    return conditions


def saved_features(directory, *, splits=()):
    """Return features, conditions and provenance with the source path and NPZ checksum.

    Feature keys h_<split> have shape (samples, features); y_<split> holds
    matching labels. Requested splits must have been explicitly saved.
    """
    path = Path(directory)
    conditions = _conditions(path)
    if not conditions["save_features"] and not conditions["save_full_test"]:
        raise ValueError("Features were intentionally not saved; re-extract them from saved encoder weights")
    content = (path / "features.npz").read_bytes()
    with np.load(BytesIO(content), allow_pickle=False) as source:
        features = {key: source[key] for key in source.files}
    validate_features(features, conditions.get("saved_feature_splits", []))

    for split in splits:
        enabled = conditions["save_full_test"] if split == "test_full" else conditions["save_features"]
        if not enabled:
            label = "full-test (test_full)" if split == "test_full" else split
            raise ValueError(f"{label} features were intentionally not saved; re-extract them from saved encoder weights")
    validate_features(features, splits)
    provenance = {"path": str(path.resolve()), "features_sha256": hashlib.sha256(content).hexdigest()}
    return features, conditions, provenance


def readout_only(source, config, directory, *, seed=None, device="cpu", script_path=None,
                 settings_path=None, init_stream=None, shuffle_stream=None, save_readout_history=True):
    """Train and save a new classifier on saved features; return its result.

    Omitted seed and stream names reuse the source records. This trains only
    the readout and writes to a new directory, leaving the encoder unchanged.
    """
    if type(save_readout_history) is not bool:
        raise ValueError("save_readout_history must be True or False")
    splits = ["train", "val", "test"] + (["test_full"] if config.readout_final_eval_full_test else [])
    features, previous, provenance = saved_features(source, splits=splits)
    seed = previous["spec"]["seed_index"] if seed is None else seed
    streams = previous["readout_streams"]
    init_stream, shuffle_stream = init_stream or streams["init"], shuffle_stream or streams["shuffle"]

    result, seeds = fit_readout(
        features, config, seed, device=device,
        init_stream=init_stream, shuffle_stream=shuffle_stream,
    )

    conditions = {
        "representation": "readout_only", "source": provenance, "input": previous["input"],
        "source_representation": previous["representation"], "spec": {"seed_index": seed},
        "config": asdict(config), "device": str(device), "seeds": seeds,
        "readout_streams": {"init": init_stream, "shuffle": shuffle_stream},
        "save_readout_history": save_readout_history,
    }
    write_result(directory, conditions, tables=readout_tables(result, save_history=save_readout_history),
                 script_path=script_path, settings_path=settings_path)
    return result


def analyse_saved(source, directory, *, analysis=AnalysisSettings(), script_path=None, settings_path=None):
    """Compute, save and return entropy/silhouette tables from saved features."""
    features, previous, provenance = saved_features(source, splits=[analysis.split])
    entropy, silhouette = analyse_features(features, analysis)

    write_result(
        directory, {
            "representation": "analysis", "source": provenance, "input": previous["input"],
            "analysis": asdict(analysis), "spec": previous["spec"],
        },
        tables={"entropy": entropy, "silhouette": silhouette},
        script_path=script_path, settings_path=settings_path,
    )
    return entropy, silhouette


def embed_saved(source, directory, params, *, split="test", seed=None, script_path=None, settings_path=None):
    """Save a UMAP embedding with labels and return its (samples, components) points.

    The UMAP random state comes from the named stream of the supplied seed,
    or the source seed when omitted. No encoder inference is performed.
    """
    from .umap_embed import embed

    features, previous, provenance = saved_features(source, splits=[split])
    seed = previous["spec"]["seed_index"] if seed is None else seed
    random_state = int(stream(seed, "umap").integers(0, 2**31))
    points = embed(features[f"h_{split}"], params, random_state=random_state)

    write_result(
        directory, {
            "representation": "umap", "source": provenance, "input": previous["input"],
            "spec": {"seed_index": seed},
            "umap": {
                **asdict(params), "split": split, "random_state": random_state,
                "version": importlib.metadata.version("umap-learn"),
            },
        },
        arrays={"embedding": {"embedding": points, "labels": features[f"y_{split}"]}},
        script_path=script_path, settings_path=settings_path,
    )
    return points


def features_from_encoder(source, dataset_path, config, *, include_full_test=True, device="cpu"):
    """Extract raw inputs or saved sparse/MLP features without training or saving.

    Return h_<split> arrays of shape (samples, features) and matching y_<split>
    labels for train/val/test, plus test_full when requested. The supplied config
    controls sparse inference; the caller supplies the dataset to encode.
    """
    import torch
    from .data import load_dataset
    from .encoders import sparse_features, mlp_frozen_features, raw_features, labels_dict
    from .model.params import ModelParams
    from .mlp import OneHiddenMLP

    previous = _conditions(source)
    dataset = load_dataset(dataset_path)
    representation = previous["representation"]
    filename = "weights_trained" if representation == "sparse_trained" else "encoder_weights"

    if representation == "raw":
        values = raw_features(dataset.arrays, include_full_test=include_full_test)
    else:
        if not previous["save_encoder_weights"]:
            raise ValueError("Encoder weights were intentionally not saved; feature re-extraction is unavailable")
        with np.load(Path(source) / f"{filename}.npz", allow_pickle=False) as stored:
            weights = {key: stored[key] for key in stored.files}
        if representation in ("sparse_initial", "sparse_trained"):
            model = ModelParams.from_arrays(previous["spec"]["model_code"], weights, torch.device(device))
            with torch.inference_mode():
                values = sparse_features(
                    model, dataset.arrays, config, torch.device(device),
                    include_full_test=include_full_test,
                )
        elif representation == "mlp_frozen":
            hidden, n_in = weights["hidden.weight"].shape
            with torch.random.fork_rng(devices=[]):
                model = OneHiddenMLP(n_in, hidden, weights["readout.weight"].shape[0])
            model.load_state_dict({key: torch.as_tensor(value) for key, value in weights.items()})
            model.to(torch.device(device)).eval()
            with torch.no_grad():
                values = mlp_frozen_features(
                    model, dataset.arrays, torch.device(device),
                    include_full_test=include_full_test,
                )
        else:
            raise ValueError("This result does not contain a reusable encoder")

    values.update(labels_dict(dataset.arrays, include_full_test=include_full_test))
    validate_features(values, ["train", "val", "test"] + (["test_full"] if include_full_test else []))
    return values


def compare_tables(directories, table="summary"):
    """Join saved rows with source/spec/config columns after checking comparability.

    Require a shared dataset checksum and the measurement settings relevant to
    the selected table. Rows remain unaggregated.
    """
    if table not in ("summary", "history", "entropy", "silhouette", "monitor"):
        raise ValueError("Choose an existing numerical table")

    frames, definition = [], None
    for directory in directories:
        conditions = _conditions(directory)
        if table == "history" and not conditions["save_readout_history"]:
            raise ValueError(f"Readout history was intentionally not saved: {directory}")
        dataset = conditions.get("input", {}).get("sha256")
        if not dataset:
            raise ValueError("Dataset identity is unknown; compare explicitly outside this helper")
        if table in ("entropy", "silhouette"):
            fields = (
                ("split", "population_bins", "lifetime_bins", "allow_dropped")
                if table == "entropy" else ("split", "metric", "center")
            )
            settings = {key: conditions["analysis"][key] for key in fields}
        else:
            fields = (
                ("monitor_probe_split", "monitor_var_weight")
                if table == "monitor" else ("readout_standardize", "readout_final_eval_full_test")
            )
            settings = {key: conditions["config"][key] for key in fields}
        current = json.dumps([dataset, settings], sort_keys=True)
        if definition is not None and current != definition:
            raise ValueError("Dataset or measurement definitions differ; keep these tables separate")
        definition = current

        frame = pd.read_csv(Path(directory) / f"{table}.csv", float_precision="round_trip")
        frame["source_directory"] = str(Path(directory))
        frame["representation"] = conditions["representation"]
        for key, value in conditions.get("spec", {}).items():
            if key not in frame:
                frame[key] = value
        for key, value in conditions.get("config", {}).items():
            frame[f"config.{key}"] = [value] * len(frame)
        frames.append(frame)
    if not frames:
        raise ValueError("Choose at least one result")
    return pd.concat(frames, ignore_index=True)


def target_evidence(directories):
    """Return per-run target evidence and candidate diagnostics without saving a choice.

    Inputs must be unique trained sparse model/target/seed runs. Candidate
    selection uses the final classifier validation loss; other criteria remain
    diagnostics.
    """
    from .targets import _argmax_table, TIEBREAK_RULE, TARGET_CRITERION, classifier_target_candidates

    directories = list(directories)
    for table in ("summary", "silhouette", "monitor"):
        compare_tables(directories, table)

    rows = []
    for directory in directories:
        conditions = _conditions(directory)
        if conditions["representation"] != "sparse_trained":
            raise ValueError("Target diagnostics need trained sparse results")
        summary = pd.read_csv(Path(directory) / "summary.csv", float_precision="round_trip")
        silhouette = pd.read_csv(Path(directory) / "silhouette.csv", float_precision="round_trip")
        monitor = pd.read_csv(Path(directory) / "monitor.csv", float_precision="round_trip")
        if len(summary) != 1 or len(silhouette) != 1 or monitor.empty:
            raise ValueError("Incomplete target diagnostic inputs")
        if "final_val_loss" not in summary:
            raise ValueError("Target selection requires saved final_val_loss; legacy loss is not reconstructed")
        rows.append({
            **conditions["spec"], "final_val_accuracy": summary.final_val_accuracy.iloc[0],
            "final_val_loss": summary.final_val_loss.iloc[0],
            "silhouette": silhouette.silhouette.iloc[0],
            "composite": monitor.sort_values("epoch").composite.iloc[-1],
        })

    frame = pd.DataFrame(rows)
    if frame.empty or frame.duplicated(["model_code", "targ", "seed_index"]).any():
        raise ValueError("Target evidence must contain unique model/target/seed rows")
    if not np.isfinite(frame[["final_val_accuracy", "composite"]]).all().all():
        raise ValueError("Target evidence requires finite accuracy and monitor loss")

    criteria = {"median_final_val_accuracy": _argmax_table(frame, "final_val_accuracy"),
                "median_composite_loss_min": _argmax_table(frame, "composite", maximize=False)}
    candidates = classifier_target_candidates(frame)
    criteria[TARGET_CRITERION] = _argmax_table(frame, "final_val_loss", maximize=False)
    missing = int(frame.silhouette.isna().sum())
    if not missing:
        criteria["median_trained_silhouette"] = _argmax_table(frame, "silhouette")
    return frame, {"criteria": criteria, "selection_criterion": TARGET_CRITERION,
                   "target_candidates": candidates.to_dict("records"),
                   "undefined_silhouette_rows": missing, "tiebreak_rule": TIEBREAK_RULE}
