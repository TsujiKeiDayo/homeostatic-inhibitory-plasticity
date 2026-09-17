"""Save and load experiment arrays, tables and provenance as NPZ/CSV/JSON."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.metadata
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile

import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True, kw_only=True)
class SaveOptions:
    """Storage choices only; these never control scientific computation."""

    save_features: bool = True
    save_full_test: bool = False  # Preserve the low-level API's storage default.
    save_encoder_weights: bool = True
    save_activity: bool = True
    save_summaries: bool = True
    save_readout_history: bool = True

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) is not bool:
                raise ValueError(f"{name} must be True or False")


def recorded_options_issues(conditions):
    """Return missing or invalid saving switches and readout stream names."""
    representation = conditions.get("representation")
    if representation in ("sparse_trained", "sparse_initial", "raw", "mlp_frozen"):
        keys = SaveOptions.__dataclass_fields__
    elif representation == "readout_only":
        keys = ("save_readout_history",)
    else:
        return []  # HP, metrics and figure tables have different output contracts.
    errors = [
        f"conditions.{key}: missing or not bool" for key in keys
        if type(conditions.get(key)) is not bool
    ]
    streams = conditions.get("readout_streams")
    if (
        not isinstance(streams, dict)
        or any(
            not isinstance(streams.get(key), str) or not streams[key]
            for key in ("init", "shuffle")
        )
    ):
        errors.append("conditions.readout_streams: recorded init/shuffle names required")
    return errors


def single_files(conditions):
    """Return NPZ and CSV stems promised by explicit single-run saving choices."""
    issues = recorded_options_issues(conditions)
    if issues:
        raise ValueError(f"Invalid saved conditions: {issues}")

    representation = conditions["representation"]
    arrays = []
    if conditions["save_features"] or conditions["save_full_test"]:
        arrays.append("features")
    if conditions["save_encoder_weights"]:
        if representation == "sparse_trained":
            arrays.extend(("weights_initial", "weights_trained"))
        elif representation in ("sparse_initial", "mlp_frozen"):
            arrays.append("encoder_weights")

    tables = ["summary", "entropy", "silhouette"]
    if conditions["save_readout_history"]:
        tables.append("history")
    if representation == "sparse_trained":
        tables.append("monitor")
        for name in ("summaries", "activity"):
            if conditions[f"save_{name}"]:
                arrays.append(name)
    return arrays, tables


def _atomic(path, write):
    """Write through a temporary sibling so partial content never replaces the target."""
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".writing-", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        write(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _json(path, value):
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
    _atomic(path, lambda temporary: temporary.write_text(text, encoding="utf-8"))


def _npz(path, arrays):
    def write(temporary):
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
    _atomic(path, write)


def _environment():
    root = Path(__file__).resolve().parents[2]
    commit, dirty = None, None
    try:
        revision = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, timeout=5)
        status = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True, timeout=5)
        commit = revision.stdout.decode().strip() if revision.returncode == 0 else None
        dirty = bool(status.stdout) if status.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        pass  # Git identification is optional; never invent a revision.

    # Includes uncommitted/untracked Python source, unlike HEAD + a dirty flag.
    source_files = sorted((root / "src/hnn2").rglob("*.py"))
    source_hashes = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in source_files}
    return {
        "python": platform.python_version(), "torch_threads": torch.get_num_threads(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "torch", "scikit-learn")
        },
        "source_root": str(root), "git_commit": commit, "git_dirty": dirty,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "torch_cuda_version": torch.version.cuda,
        "source_sha256": source_hashes,
    }


def save_single(result, directory: str | Path, *, script_path: str | Path | None = None,
                settings_path: str | Path | None = None, saving: SaveOptions = SaveOptions(),
                resume: bool = False) -> Path:
    """Write a new condition directory; COMPLETE is written last.

    Full-test evaluation stays in summary.csv even when its large feature array
    is not saved. saving.save_features controls train/val/test, independently of
    saving.save_full_test. Only already computed arrays are saved.
    Complete directories are never overwritten. resume=True restarts an
    incomplete directory only when its scientific and saving conditions agree.
    """
    adapted = result.adaptation
    features = {
        key: value for key, value in result.features.items()
        if (saving.save_full_test if key in ("h_test_full", "y_test_full") else saving.save_features)
    }
    if saving.save_full_test and not all(key in features for key in ("h_test_full", "y_test_full")):
        raise ValueError("Full-test features were not computed; cannot save them. "
                         "Set save_full_test=False or explicitly enable full-test computation.")

    # Assemble the requested arrays with explicit sample order for later checks.
    arrays = {"features": features} if features else {}
    for key in list(features):
        if key.startswith("h_"):
            features[f"sample_index_{key[2:]}"] = np.arange(len(features[key]), dtype=np.int64)
    if saving.save_encoder_weights and result.encoder_weights is not None:
        arrays["encoder_weights"] = result.encoder_weights
    if adapted is not None:
        if saving.save_encoder_weights:
            arrays.update(weights_initial=adapted.params_initial.to_arrays(),
                          weights_trained=adapted.params_trained.to_arrays())
        if saving.save_summaries:
            arrays["summaries"] = adapted.summaries_arrays()
        if saving.save_activity:
            activity = {f"o_E_ep{epoch:02d}": value for epoch, value in adapted.analysis_activity.items()}
            activity["labels"] = result.features[f"y_{result.conditions['config']['analysis_probe_split']}"]
            activity["sample_index"] = np.arange(len(activity["labels"]), dtype=np.int64)
            arrays["activity"] = activity

    tables = {**readout_tables(result.readout, save_history=saving.save_readout_history),
              "entropy": result.entropy, "silhouette": result.silhouette}
    if adapted is not None:
        tables["monitor"] = pd.DataFrame(adapted.monitor_records)

    conditions = {**result.conditions, **asdict(saving),
                  "saved_feature_splits": [key[2:] for key in features if key.startswith("h_")]}
    if conditions.get("output_version") == 1:
        from .artifact_checks import output_issues
        issues = output_issues(conditions, arrays, tables)
        if issues:
            raise ValueError(f"Cannot save current output: {issues}")

    return write_result(directory, conditions, arrays=arrays, tables=tables,
                        script_path=script_path, settings_path=settings_path, resume=resume)


def readout_tables(result, *, save_history=True):
    """Return a one-row classifier summary and optional per-step history."""
    summary = {key: value for key, value in asdict(result).items() if key != "history"}
    summary["final_cumulative_l1"] = result.history.cumulative_l1[-1]
    tables = {"summary": pd.DataFrame([summary])}
    if save_history:
        history = asdict(result.history)
        if not history["val_evaluated"]:  # Old in-memory fixtures remain readable as legacy.
            history.pop("val_evaluated")
        tables["history"] = pd.DataFrame(history)
    return tables


def _same_conditions(saved, expected):
    issues = recorded_options_issues(saved) + recorded_options_issues(expected)
    if issues:
        raise ValueError(f"Conditions differ or are incomplete: {issues}")
    keys = ("spec", "config", "analysis", "input", "representation", "device", "mlp",
            "source", "selection", "sweep_specs", "umap", "training_epochs",
            "readout_streams", *SaveOptions.__dataclass_fields__)
    if saved.get("output_version") == expected.get("output_version") == 1:
        keys += ("sample_order", "numerics", "rng", "axes")
    canonical = lambda value: json.dumps(value, sort_keys=True, allow_nan=False)
    return all(canonical(saved.get(k)) == canonical(expected.get(k)) for k in keys)


def completed(directory, conditions, required):
    """Return whether matching conditions are complete and required files exist.

    Incomplete matching results return False. Mismatched conditions or missing
    required files in completed results raise. When the requested output_version
    is 1, saved results must also pass current-format content validation.
    """
    path = Path(directory)
    if not (path / "COMPLETE").exists():
        if (path / "conditions.json").exists():
            saved = json.loads((path / "conditions.json").read_text(encoding="utf-8"))
            if not _same_conditions(saved, conditions):
                raise ValueError(f"Conditions differ at incomplete result {path}; choose a new output directory")
        return False

    saved = json.loads((path / "conditions.json").read_text(encoding="utf-8"))
    if not _same_conditions(saved, conditions):
        raise ValueError(f"Conditions differ at {path}; choose a new output directory")
    if conditions.get("output_version") == 1:
        if saved.get("output_version") != 1:
            raise ValueError(f"Legacy COMPLETE lacks current figure outputs: {path}; choose a new output directory")
        from .artifact_checks import output_issues
        actual, arrays, tables = read_single(path)
        issues = output_issues(actual, arrays, tables)
        if issues:
            raise ValueError(f"Completed result has invalid current outputs: {issues}")

    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise ValueError(f"Completed result lacks {missing}: {path}")
    return True


def write_result(directory, conditions, *, arrays=None, tables=None, documents=None,
                 script_path=None, settings_path=None, resume=False):
    """Write named NPZ/CSV/JSON outputs and provenance; return the resolved path.

    Each file is replaced atomically, with COMPLETE written last. Existing
    directories require resume=True and matching incomplete conditions.
    Failures are recorded in FAILED.json when possible.
    """
    arrays, tables, documents = arrays or {}, tables or {}, documents or {}
    script = Path(script_path).read_bytes() if script_path is not None else None
    script_name = (
        "experiment.ipynb"
        if script_path is not None and Path(script_path).suffix == ".ipynb"
        else "experiment.py"
    )
    settings = Path(settings_path).read_bytes() if settings_path is not None else None

    path = Path(directory).expanduser().resolve()
    if path.exists():
        if not resume or (path / "COMPLETE").exists():
            raise FileExistsError(path)
        saved = json.loads((path / "conditions.json").read_text(encoding="utf-8"))
        if not _same_conditions(saved, conditions):
            raise ValueError(f"Refusing to resume different conditions: {path}")
    else:
        path.mkdir(parents=True)

    try:
        _json(path / "conditions.json", {
            **conditions, "environment": _environment(),
            **({"settings": "experiment_settings.py"} if settings is not None else {}),
            "script": script_name if script is not None else None,
        })
        for name, values in arrays.items():
            _npz(path / f"{name}.npz", values)
        for name, table in tables.items():
            _atomic(path / f"{name}.csv", lambda temporary, table=table: table.to_csv(temporary, index=False))
        for name, value in documents.items():
            _json(path / f"{name}.json", finite_json(value))
        if script is not None:
            _atomic(path / script_name, lambda temporary: temporary.write_bytes(script))
        if settings is not None:
            _atomic(path / "experiment_settings.py", lambda temporary: temporary.write_bytes(settings))

        (path / "FAILED.json").unlink(missing_ok=True)
        _atomic(path / "COMPLETE", lambda temporary: temporary.write_text("complete\n", encoding="utf-8"))
    except Exception as exc:
        try:
            _json(path / "FAILED.json", {"error": type(exc).__name__, "message": str(exc)})
        except OSError:
            pass  # Keep the original failure, including when the disk is full.
        raise
    return path


def read_single(directory: str | Path):
    """Load a completed result as (conditions, named NPZ contents, named tables).

    Require the files promised by the recorded saving choices; do not infer
    missing choices or recompute missing outputs.
    """
    path = Path(directory)
    if not (path / "COMPLETE").is_file():
        raise ValueError("Incomplete single result: COMPLETE missing")
    conditions = json.loads((path / "conditions.json").read_text(encoding="utf-8"))
    array_names, table_names = single_files(conditions)
    required = [*[f"{n}.npz" for n in array_names], *[f"{n}.csv" for n in table_names]]
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise ValueError(f"Incomplete single result: {missing}")

    arrays = {}
    for name in array_names:
        with np.load(path / f"{name}.npz", allow_pickle=False) as source:
            arrays[name] = {key: source[key] for key in source.files}
    tables = {name: pd.read_csv(path / f"{name}.csv", float_precision="round_trip") for name in table_names}
    return conditions, arrays, tables


def finite_json(value):
    """Undefined diagnostic values remain null in JSON, never zero/success."""
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    return None if isinstance(value, float) and not np.isfinite(value) else value


def record_failure(directory, conditions, error):
    """Record conditions and FAILED.json without overwriting a complete result."""
    path = Path(directory)
    if (path / "COMPLETE").exists():
        raise FileExistsError(path)
    path.mkdir(parents=True, exist_ok=True)
    if (path / "conditions.json").exists():
        saved = json.loads((path / "conditions.json").read_text(encoding="utf-8"))
        if not _same_conditions(saved, conditions):
            raise ValueError(f"Failure directory belongs to different conditions: {path}")

    _json(path / "conditions.json", conditions)
    _json(path / "FAILED.json", {"error": type(error).__name__, "message": str(error)})
