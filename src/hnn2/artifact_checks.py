"""Check saved experiment inputs to figures. No training, inference or repair."""
import json
import numpy as np
import pandas as pd
from zipfile import BadZipFile

from .result_io import recorded_options_issues


def _table(tables, name, columns, errors, length=None):
    value = tables.get(name)
    if value is None:
        errors.append(f"missing table {name}")
        return None
    missing = set(columns) - set(value.columns)
    if missing or (length is not None and len(value) != length):
        errors.append(f"{name}: missing columns {sorted(missing)} or incorrect row count")
        return None
    return value


def readout_issues(c, tables, *, require_history=False):
    """Return classifier summary/history issues under the recorded conditions.

    require_history=True also rejects intentionally omitted history for RF-05.
    """
    errors = recorded_options_issues(c)
    if errors:
        return errors
    if c.get("output_version") != 1:
        return ["legacy output: current readout observations are not certified"]

    # Reconstruct the batch and validation schedules from the saved budget.
    config, shapes = c["config"], c["input"]["shapes"]
    for key in ("batch_size", "readout_epochs", "readout_eval_every"):
        if type(config[key]) is not int or config[key] < 1:
            return [f"config.{key}: expected a positive integer"]
    ntrain = shapes["y_train"][0]
    batches = (ntrain + config["batch_size"] - 1) // config["batch_size"]
    steps = batches * config["readout_epochs"]
    if steps < 1:
        return ["readout training split is empty"]

    summary = _table(
        tables, "summary", [
            "final_val_loss", "final_step", "final_val_accuracy", "best_val_accuracy",
            "final_test_accuracy", "final_test_accuracy_full", "n_features", "n_classes",
            "final_cumulative_l1",
        ], errors, 1,
    )
    if summary is not None:
        s = pd.to_numeric(summary.iloc[0], errors="coerce")
        for key in ("final_val_loss", "final_val_accuracy", "best_val_accuracy", "final_test_accuracy",
                    "final_cumulative_l1", "n_features", "n_classes"):
            if not np.isfinite(s[key]) or s[key] < 0:
                errors.append(f"summary.{key}: missing/nonfinite/negative")
        for key in ("final_val_accuracy", "best_val_accuracy", "final_test_accuracy"):
            if s[key] > 1:
                errors.append(f"summary.{key}: outside [0,1]")
        width = shapes["x_train_big"][1] if c["representation"] == "raw" else config["n_excitatory"]
        if (
            s.n_features != width or not np.isfinite(s.n_classes)
            or s.n_classes < 1 or s.n_classes != int(s.n_classes)
        ):
            errors.append("summary: feature/class count differs from encoder or is not a positive integer")
        if s.final_step != steps:
            errors.append("summary.final_step differs from completed training budget")
        if s.best_val_accuracy < s.final_val_accuracy:
            errors.append("summary.best_val_accuracy excludes final evaluation")
        if config["readout_final_eval_full_test"]:
            if not np.isfinite(s.final_test_accuracy_full) or not 0 <= s.final_test_accuracy_full <= 1:
                errors.append("summary.final_test_accuracy_full missing/invalid")
        elif not pd.isna(s.final_test_accuracy_full):
            errors.append("summary.final_test_accuracy_full present despite disabled evaluation")

    if not c["save_readout_history"]:
        if require_history:
            errors.append("save_readout_history=False: RF-05 observations unavailable")
        return errors
    h = _table(
        tables, "history", [
            "step", "epoch", "batch_loss", "batch_accuracy", "val_loss", "val_accuracy",
            "l1_delta", "cumulative_l1", "val_evaluated",
        ], errors, steps,
    )
    if h is not None:
        measured = np.arange(1, steps + 1) % config["readout_eval_every"] == 0
        if not np.array_equal(h.step, np.arange(1, steps + 1)):
            errors.append("history.step not contiguous")
        if not np.array_equal(h.epoch, np.repeat(np.arange(config["readout_epochs"]), batches)):
            errors.append("history.epoch differs from batch schedule")
        if h.val_evaluated.dtype.kind != "b" or not np.array_equal(h.val_evaluated, measured):
            errors.append("history.val_evaluated differs from recorded evaluation schedule")

        numeric = h[["batch_loss", "batch_accuracy", "l1_delta", "cumulative_l1"]].to_numpy(float)
        if not np.isfinite(numeric).all() or (numeric < 0).any() or (h.batch_accuracy > 1).any():
            errors.append("history: invalid batch/L1 values")
        if not np.allclose(h.cumulative_l1, h.l1_delta.cumsum(), rtol=1e-7, atol=1e-9):
            errors.append("history.cumulative_l1 differs from summed increments")

        # Between evaluations, history must repeat the latest measured value.
        for key in ("val_loss", "val_accuracy"):
            actual = h[key].to_numpy(float)
            expected = h[key].where(measured).ffill().to_numpy(float)
            if (
                not np.isfinite(actual[measured]).all() or (actual[measured] < 0).any()
                or not np.allclose(actual, expected, equal_nan=True)
            ):
                errors.append(f"history.{key}: invalid measured/repeated values")
        if (h.val_accuracy[measured] > 1).any():
            errors.append("history.val_accuracy outside [0,1]")

        if summary is not None:
            if not np.isclose(h.cumulative_l1.iloc[-1], s.final_cumulative_l1, rtol=1e-7, atol=1e-9):
                errors.append("history/summary cumulative_l1 disagree")
            best = max([s.final_val_accuracy, *h.val_accuracy[measured]])
            if not np.isclose(s.best_val_accuracy, best, rtol=1e-7, atol=1e-9):
                errors.append("summary.best_val_accuracy differs from measured/final maximum")
            if measured[-1]:
                for key, final in (("val_accuracy", "final_val_accuracy"), ("val_loss", "final_val_loss")):
                    if not np.isclose(h[key].iloc[-1], s[final], rtol=1e-7, atol=1e-9):
                        errors.append(f"history/summary final {key} disagree at same step")
    return errors


def output_issues(c, arrays, tables, *, require_all=False, dataset=None):
    """Return explicit shortages without filling missing condition records.

    With require_all=False, check only the files promised by saving switches.
    require_all=True checks all raw inputs of the adopted figures. Passing a
    loaded Dataset additionally checks source identity and sample/label pairing.
    No missing value is inferred and no source file is changed.
    """
    errors = recorded_options_issues(c)
    if errors:
        return errors
    if c.get("output_version") != 1:
        return ["legacy output: current figure inputs are not certified (do not overwrite)"]

    # Check the recorded reconstruction protocol before validating its outputs.
    config, rep = c["config"], c["representation"]
    trained = rep == "sparse_trained"
    from .config import RunSpec, experiment_config_from_json
    from .single import reconstruction_conditions
    spec = RunSpec(c["spec"].get("model_code", "rec"), c["spec"].get("targ", 1.),
                   c["spec"].get("eta", 1.), c["spec"]["seed_index"])
    expected = reconstruction_conditions(spec, experiment_config_from_json(json.dumps(config)), rep)
    for key, value in expected.items():
        if c.get(key) != value:
            errors.append(f"conditions.{key} missing or differs from reconstruction protocol")
    errors.extend(readout_issues(c, tables))
    if any(
        type(config[key]) is not int or config[key] < 1
        for key in ("batch_size", "n_epochs", "n_excitatory", "n_inhibitory")
    ):
        return errors + ["invalid adaptation dimensions or budget"]

    def array(values, key, shape, dtype):
        value = values.get(key)
        if value is None:
            errors.append(f"missing array {key}")
            return None
        if value.shape != tuple(shape) or value.dtype != np.dtype(dtype):
            errors.append(f"{key}: expected {dtype}{tuple(shape)}, got {value.dtype}{value.shape}")
            return None
        if not np.isfinite(value).all():
            errors.append(f"{key}: nonfinite values")
        return value

    def table(name, columns, length=None):
        return _table(tables, name, columns, errors, length)

    shapes = c["input"]["shapes"]
    if dataset is not None and dataset.source["sha256"] != c["input"]["sha256"]:
        errors.append("input checksum differs")
    if dataset is not None and dataset.source["shapes"] != shapes:
        errors.append("input shapes differ from recorded source")

    splits = ["train", "val", "test"] if c["save_features"] else []
    if c["save_full_test"]:
        splits.append("test_full")
    if sorted(c.get("saved_feature_splits", [])) != sorted(splits):
        errors.append("saved_feature_splits differs from saving switches")
    if require_all:
        for key in ("save_features", "save_encoder_weights", "save_readout_history"):
            if key == "save_encoder_weights" and rep == "raw":
                continue
            if not c[key]:
                errors.append(f"{key}=False: figure input unavailable")
        if config["readout_final_eval_full_test"] and not c["save_full_test"]:
            errors.append("save_full_test=False: full-test figure input unavailable")
        if trained:
            for key in ("save_summaries", "save_activity"):
                if not c[key]:
                    errors.append(f"{key}=False: figure input unavailable")

    # Feature rows must retain the input split's sample and label order.
    ntrain = shapes["y_train"][0]
    batches = (ntrain + config["batch_size"] - 1) // config["batch_size"]
    width = shapes["x_train_big"][1] if rep == "raw" else config["n_excitatory"]
    for split in splits:
        n = shapes[f"y_{split}"][0]
        features = arrays.get("features", {})
        array(features, f"h_{split}", (n, width), "float32")
        y = array(features, f"y_{split}", (n,), "int64")
        indices = array(features, f"sample_index_{split}", (n,), "int64")
        if indices is not None and not np.array_equal(indices, np.arange(n)):
            errors.append(f"{split}: sample order differs from input split rows")
        if (
            dataset is not None and y is not None
            and not np.array_equal(y, dataset.arrays[f"y_{split}"])
        ):
            errors.append(f"{split}: labels differ from input rows")

    if c["save_encoder_weights"] and rep != "raw":
        if rep in ("sparse_initial", "sparse_trained"):
            names = ["weights_initial", "weights_trained"] if trained else ["encoder_weights"]
            model, ne, ni = c["spec"]["model_code"], config["n_excitatory"], config["n_inhibitory"]
            for name in names:
                weights = arrays.get(name, {})
                array(weights, "w_EIn", (ne, shapes["x_train_small"][1]), "float32")
                array(weights, "theta", (ne,), "float32")
                if model != "thresh":
                    array(weights, "w_EI", (ne, ni), "float32")
                    key, shape = (
                        ("w_IE", (ni, ne)) if model == "rec"
                        else ("w_IIn", (ni, shapes["x_train_small"][1]))
                    )
                    array(weights, key, shape, "float32")
                elif any(key in weights for key in ("w_EI", "w_IE", "w_IIn")):
                    errors.append("thresh: inhibitory weights must be absent")
        else:
            weights = arrays.get("encoder_weights", {})
            ne = config["n_excitatory"]
            array(weights, "hidden.weight", (ne, shapes["x_train_small"][1]), "float32")
            array(weights, "hidden.bias", (ne,), "float32")
            if not {"readout.weight", "readout.bias"} <= weights.keys():
                errors.append("MLP encoder weights missing")

    if trained:
        epochs, ne = config["n_epochs"], config["n_excitatory"]
        monitor = table(
            "monitor", [
                "epoch", "eta", "composite", "l_mean", "rate_var", "rate_mean", "rate_wmean",
                "rate_mean_train", "rate_wmean_train",
            ], epochs,
        )
        if monitor is not None:
            if not np.array_equal(monitor.epoch, np.arange(epochs)):
                errors.append("monitor.epoch differs from adaptation schedule")
            if not np.isfinite(monitor.to_numpy(float)).all():
                errors.append("monitor: nonfinite values")

        if c["save_summaries"]:
            # Trajectory row 0 precedes adaptation; each later row follows a batch update.
            q = arrays.get("summaries", {})
            k = ne if c["spec"]["model_code"] == "thresh" else ne * config["n_inhibitory"]
            p = array(q, "param_trajectory", (1 + epochs * batches, k), "float32")
            es = array(q, "trajectory_epoch", (1 + epochs * batches,), "int64")
            bs = array(q, "trajectory_batch_start", (1 + epochs * batches,), "int64")
            if es is not None and not np.array_equal(es, np.r_[-1, np.repeat(np.arange(epochs), batches)]):
                errors.append("trajectory_epoch: initial/batch alignment differs")
            if bs is not None and not np.array_equal(
                bs, np.r_[-1, np.tile(np.arange(0, ntrain, config["batch_size"]), epochs)]
            ):
                errors.append("trajectory_batch_start: initial/batch alignment differs")
            key = "theta" if c["spec"]["model_code"] == "thresh" else "w_EI"
            if p is not None:
                for name, idx in (("weights_initial", 0), ("weights_trained", -1)):
                    if (
                        name in arrays and key in arrays[name]
                        and not np.array_equal(p[idx], arrays[name][key].ravel())
                    ):
                        errors.append(f"trajectory endpoint differs from {name}")

            n = shapes[f"y_{config['analysis_probe_split']}"][0]
            from .metrics.bins import POPULATION_BINS, LIFETIME_BINS
            for epoch in {0, epochs - 1}:
                for axis, length, bins in (("pop", n, POPULATION_BINS), ("life", ne, LIFETIME_BINS)):
                    for metric in ("r", "active_frac", "l1", "l2"):
                        array(q, f"probe_ep{epoch:02d}_{metric}_{axis}", (length,), "float32")
                    array(q, f"probe_ep{epoch:02d}_hist_{axis}", (bins.n_bins,), "int64")
                array(q, f"probe_ep{epoch:02d}_dropped_mass", (2,), "float32")

        if c["save_activity"]:
            a = arrays.get("activity", {})
            split = config["analysis_probe_split"]
            n = shapes[f"y_{split}"][0]
            labels = array(a, "labels", (n,), "int64")
            indices = array(a, "sample_index", (n,), "int64")
            if indices is not None and not np.array_equal(indices, np.arange(n)):
                errors.append("activity sample order differs")
            if (
                dataset is not None and labels is not None
                and not np.array_equal(labels, dataset.arrays[f"y_{split}"])
            ):
                errors.append("activity labels differ from input")
            for epoch in {0, epochs - 1}:
                array(a, f"o_E_ep{epoch:02d}", (ne, n), "float32")

    # Metric rows must use the recorded analysis split, axes and definitions.
    for name, columns in (
        ("entropy", [
            "split", "axis", "H", "Hhat", "S", "S_prime", "zero_bin_frac",
            "dropped_mass", "lo", "hi", "width", "n_bins",
        ]),
        ("silhouette", ["split", "silhouette", "metric", "center"]),
    ):
        frame = table(name, columns)
        if frame is None:
            continue
        if (require_all or name in c["analysis"]["metrics"]) and frame.empty:
            errors.append(f"{name} metric was not computed")
        elif not frame.empty:
            if not frame["split"].eq(c["analysis"]["split"]).all():
                errors.append(f"{name}: split differs from conditions")
            if name == "silhouette" and (
                len(frame) != 1
                or not frame.metric.eq(c["analysis"]["metric"]).all()
                or not frame.center.eq(c["analysis"]["center"]).all()
            ):
                errors.append("silhouette: metric/center/rows differ from conditions")
            if name == "silhouette":
                values = frame.silhouette.to_numpy(float)
                if (np.isinf(values) | (np.abs(values) > 1)).any():
                    errors.append("silhouette: values outside [-1,1]")
            if name == "entropy":
                if len(frame) != 2 or set(frame.axis) != {"population", "lifetime"}:
                    errors.append("entropy: expected one population and one lifetime row")
                else:
                    from .metrics.bins import BinSpec
                    for row in frame.itertuples():
                        bins = BinSpec(**c["analysis"][f"{row.axis}_bins"])
                        if any(
                            getattr(row, key) != getattr(bins, key)
                            for key in ("lo", "hi", "width", "n_bins")
                        ):
                            errors.append(f"entropy.{row.axis}: bins differ from conditions")
    return errors


def inspect_saved(directory, *, require_all=True, dataset=None):
    """Return a read-only issue list, including missing, legacy or corrupt outputs."""
    from .result_io import read_single

    try:
        c, a, t = read_single(directory)
        return output_issues(c, a, t, require_all=require_all, dataset=dataset)
    except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile) as exc:
        return [f"{type(exc).__name__}: {exc}"]
