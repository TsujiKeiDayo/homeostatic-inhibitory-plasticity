"""Numerical tables for the adopted figures, using saved numbers only.

Functions return ordinary DataFrames/dicts for use in a notebook. Nothing here
starts training, adopts a target, changes a source, or generates an embedding.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import pandas as pd

from .artifact_checks import output_issues, readout_issues
from .config import experiment_config_from_json
from .data import load_dataset
from .metrics.bins import BinSpec, POPULATION_BINS, LIFETIME_BINS
from .metrics.entropy import scores_from_counts
from .metrics.silhouette import representation_silhouette
from .result_io import read_single, write_result, recorded_options_issues
from .targets import TIEBREAK_RULE, TARGET_CRITERION, classifier_target_candidates
from .workflows import read_selection, _check_training_config

THRESHOLDS = (.30, .40, .50, .60, .70, .80, .85)
KEYS = ["run", "arm", "model_code", "targ", "eta", "seed_index"]


def _identity(path, c):
    spec, rep = c["spec"], c["representation"]
    model = spec.get("model_code", "")
    arm = (
        f"{model}_trained" if rep == "sparse_trained"
        else f"{model}_initial" if rep == "sparse_initial" else rep
    )
    return {"run": str(Path(path).resolve()), "arm": arm, "model_code": model,
            "targ": spec.get("targ", np.nan), "eta": spec.get("eta", np.nan), "seed_index": spec["seed_index"]}


def _source_hashes(directory):
    """Files used to identify saved figure inputs, including their completion mark."""
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(directory).iterdir())
        if p.is_file() and (p.name == "COMPLETE" or p.suffix in (".csv", ".npz", ".json", ".py", ".ipynb"))
    }


def load_figure_runs(results_root, *, readout_only=False):
    """Return runs keyed by path and a coverage table for the full planned population.

    Each run contains (conditions, arrays, tables); failed/missing runs remain in
    the coverage table even though they are absent from the run mapping.
    A COMPLETE with incompatible conditions or broken data is reported invalid,
    never retrained. Strict consumers call require_complete before aggregating;
    RF-05 may instead count missing/invalid runs separately. readout_only=True
    loads only classifier tables, while retaining dataset identity checks and
    source-file hashes; other figure switches do not change the RF-05 denominator.
    """
    root = Path(results_root).resolve()

    # Planned runs define the comparison denominator, including missing outputs.
    frames, population_sources = [], {}
    for name in ("plasticity", "baselines"):
        path = root / name / "expected_runs.csv"
        if not path.is_file():
            raise ValueError(f"Missing expected population: {path}; no population is inferred from legacy results")
        frames.append(pd.read_csv(path, float_precision="round_trip"))
        population_sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = pd.concat(frames, ignore_index=True)
    expected["model_code"] = expected.model_code.fillna("")
    expected["run"] = expected.path.map(lambda p: str(Path(p).resolve()))
    expected["arm"] = np.where(expected.representation == "sparse_trained",
                               expected.model_code + "_trained", expected.representation)
    if expected.run.duplicated().any() or expected.duplicated(["arm", "targ", "seed_index"]).any():
        raise ValueError("Duplicate planned run coordinates")

    runs, datasets = {}, {}
    states, reasons = [], []
    for row in expected.itertuples():
        path = Path(row.run)
        if not (path / "COMPLETE").is_file():
            state = (
                "failed" if (path / "FAILED.json").exists()
                else "incomplete" if path.exists() else "missing"
            )
            states.append(state)
            reasons.append(state)
            continue
        try:
            if readout_only:
                c = json.loads((path / "conditions.json").read_text(encoding="utf-8"))
                a = {}
                issues = recorded_options_issues(c)
                if issues:
                    raise ValueError("; ".join(issues))
                names = ["summary", "history"] if c["save_readout_history"] else ["summary"]
                t = {name: pd.read_csv(path / f"{name}.csv", float_precision="round_trip") for name in names}
            else:
                c, a, t = read_single(path)

            identity = _identity(path, c)
            for key in ("arm", "model_code", "seed_index"):
                if identity[key] != getattr(row, key):
                    raise ValueError(f"planned {key} differs from conditions")
            if c["representation"] == "sparse_trained" and (
                identity["targ"] != row.targ or identity["eta"] != row.eta
            ):
                raise ValueError("planned target/eta differs from conditions")
            for key in ("readout_epochs", "readout_eval_every", "batch_size"):
                if getattr(row, key, None) != c["config"][key]:
                    raise ValueError(f"planned {key} missing or differs from conditions")

            # Load/check each immutable input once. Never trust a path alone.
            source = c["input"]["path"]
            if source not in datasets:
                datasets[source] = load_dataset(source)
            if datasets[source].source["sha256"] != c["input"]["sha256"]:
                raise ValueError("input checksum differs")
            if datasets[source].source["shapes"] != c["input"]["shapes"]:
                raise ValueError("input shapes differ from recorded source")
            issues = (
                readout_issues(c, t, require_history=True) if readout_only
                else output_issues(c, a, t, require_all=True, dataset=datasets[source])
            )
            if issues:
                raise ValueError("; ".join(issues))
            c["loaded_source_sha256"] = _source_hashes(path)
            runs[row.run] = (c, a, t)
            states.append("complete")
            reasons.append("")
        except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile) as exc:
            states.append("invalid")
            reasons.append(str(exc))
    expected["state"], expected["reason"] = states, reasons

    # Mixed valid conditions cannot be resolved by treating the first run as
    # authoritative and dropping the others from a comparison's denominator.
    definitions = {json.dumps([c["input"]["sha256"], c["config"], c["analysis"]], sort_keys=True)
                   for c, _, _ in runs.values()}
    if len(definitions) > 1:
        raise ValueError("input/config/analysis differs; compare separately with explicit conditions")
    for arm in expected.arm.unique():
        definitions = {json.dumps([runs[p][0].get("mlp"), runs[p][0].get("readout_streams")], sort_keys=True)
                       for p in expected.loc[expected.arm == arm, "run"] if p in runs}
        if len(definitions) > 1:
            raise ValueError(f"MLP/readout conditions differ between seeds: {arm}")
    expected.attrs.update(population_sources=population_sources, readout_only=readout_only)
    return runs, expected


def require_complete(expected):
    """Raise with run/state/reason records if any planned run is not complete."""
    bad = expected[expected.state != "complete"]
    if len(bad):
        raise ValueError(f"Figure inputs incomplete: {bad[['run', 'state', 'reason']].to_dict('records')}")


def selected_runs(runs, targets):
    """Filter trained runs by explicit display targets and keep all baseline runs."""
    models = {
        c["spec"]["model_code"] for c, _, _ in runs.values()
        if c["representation"] == "sparse_trained"
    }
    if set(targets) != models:
        raise ValueError("Explicit display targets required for every trained model")
    selected = {
        p: r for p, r in runs.items() if r[0]["representation"] != "sparse_trained"
        or r[0]["spec"]["targ"] == targets[r[0]["spec"]["model_code"]]
    }
    if {
        r[0]["spec"]["model_code"] for r in selected.values()
        if r[0]["representation"] == "sparse_trained"
    } != models:
        raise ValueError("Display target is absent")
    return selected


def validation_points(history, summary):
    """Return measured validation points plus the final evaluation, sorted by step.

    Repeated values between evaluations are excluded. A final point already in
    history must agree with the summary and appears only once.
    """
    if history.empty or "val_evaluated" not in history or len(summary) != 1:
        raise ValueError("missing history, evaluation flags or final summary")
    if history.val_evaluated.dtype.kind != "b" or history.step.duplicated().any():
        raise ValueError("invalid evaluation flags or duplicate history steps")

    cols = ["step", "epoch", "val_accuracy", "val_loss", "cumulative_l1", "batch_loss"]
    points = history.loc[history.val_evaluated.eq(True), cols].copy()
    points["source"] = "history"
    s = summary.iloc[0]
    if not np.isfinite([s.final_step, s.final_val_accuracy, s.final_val_loss]).all():
        raise ValueError("final evaluation missing/nonfinite")
    if s.final_step != history.step.max():
        raise ValueError("final step differs from history endpoint")
    if not np.array_equal(history.step, np.arange(1, s.final_step + 1)):
        raise ValueError("history steps are incomplete or out of order")
    if not points.empty and not np.isfinite(points[["val_accuracy", "val_loss"]]).all().all():
        raise ValueError("measured evaluation missing/nonfinite")
    if not 0 <= s.final_val_accuracy <= 1 or ((points.val_accuracy < 0) | (points.val_accuracy > 1)).any():
        raise ValueError("validation accuracy outside [0,1]")

    overlap = points.step.eq(s.final_step)
    if overlap.any():
        last = points.loc[overlap].iloc[0]
        if not np.allclose(
            [last.val_accuracy, last.val_loss], [s.final_val_accuracy, s.final_val_loss],
            rtol=1e-7, atol=1e-9,
        ):
            raise ValueError("same-step history and final evaluation disagree")
        points.loc[overlap, "source"] = "history+summary"
    else:
        last = history.loc[history.step == s.final_step].iloc[0]
        points = pd.concat([points, pd.DataFrame([{
            "step": int(s.final_step), "epoch": last.epoch,
            "val_accuracy": s.final_val_accuracy, "val_loss": s.final_val_loss,
            "cumulative_l1": s.final_cumulative_l1, "batch_loss": last.batch_loss,
            "source": "summary",
        }])], ignore_index=True)
    return points.sort_values("step", ignore_index=True)


def _display_population(expected, targets):
    """Require explicit targets and matched planned seeds before filtering."""
    if expected.run.duplicated().any() or expected.duplicated(["arm", "targ", "seed_index"]).any():
        raise ValueError("Duplicate planned run coordinates")
    models = set(expected.loc[expected.representation == "sparse_trained", "model_code"])
    if not models or set(targets) != models:
        raise ValueError("Explicit display targets required for every planned model")
    pop = expected[
        (expected.representation != "sparse_trained")
        | expected.apply(lambda r: r.model_code in targets and r.targ == targets[r.model_code], axis=1)
    ]
    arms = {*(f"{m}_{stage}" for m in models for stage in ("initial", "trained")), "raw", "mlp_frozen"}
    if set(pop.arm) != arms:
        raise ValueError("Display target or baseline is absent from planned population")
    seed_sets = pop.groupby("arm").seed_index.apply(lambda values: tuple(sorted(values)))
    if len(set(seed_sets)) != 1:
        raise ValueError("Display population requires identical planned seed sets")
    return pop


def activity_tables(runs, *, raw_bins=BinSpec(0, 6, .05), common_bins=None):
    """Return activity metric and histogram tables for AD/EN endpoints and RF-09.

    Main feature bins come from each recorded analysis definition. Probe bins
    retain the core's POPULATION/LIFETIME definitions. Common bins are a separate
    explicit diagnostic, never substituted into the main metrics.
    Features are (samples, units); saved probe activity is transposed to match.
    Population means reduce units; lifetime means reduce samples. No network
    evaluation is performed.
    """
    metrics, histograms = [], []
    for path, (c, a, _) in runs.items():
        identity = _identity(path, c)
        analysis = c["analysis"]
        feature_split = analysis["split"]
        endpoints = [(
            "initial" if c["representation"] == "sparse_initial" else "final",
            feature_split, a["features"][f"h_{feature_split}"], "features"
        )]
        if c["representation"] == "sparse_trained":
            endpoints += [(f"ep{e:02d}", c["config"]["analysis_probe_split"],
                           a["activity"][f"o_E_ep{e:02d}"].T, "activity")
                          for e in sorted({0, c["config"]["n_epochs"] - 1})]

        for stage, split, values, source in endpoints:
            axis_bins = (
                {"population": BinSpec(**analysis["population_bins"]),
                 "lifetime": BinSpec(**analysis["lifetime_bins"])}
                if source == "features" else
                {"population": POPULATION_BINS, "lifetime": LIFETIME_BINS}
            )
            for axis, reduce_axis in (("population", 1), ("lifetime", 0), ("raw", None)):
                if axis == "raw" and source != "activity":
                    continue
                data = values.ravel() if axis == "raw" else values.mean(axis=reduce_axis)
                choices = [("main", raw_bins if axis == "raw" else axis_bins[axis])]
                if common_bins is not None and axis != "raw" and source == "features":
                    choices.append(("common_range_diagnostic", common_bins[axis]))
                for scope, bins in choices:
                    allow = (
                        analysis["allow_dropped"]
                        if source == "features" and scope == "main" else True
                    )
                    counts, dropped = bins.apply(data, allow_dropped=allow)
                    scores = asdict(scores_from_counts(counts))
                    meta = {**identity, "stage": stage, "split": split, "source": source,
                            "axis": axis, "bin_scope": scope, **asdict(bins), "n_bins": bins.n_bins}
                    metrics.append({**meta, **scores, "dropped_mass": dropped,
                                    "n_values": len(data), "mean_activity": float(data.mean()),
                                    "sd_activity": float(data.std(ddof=0)),
                                    "active_fraction": float((values > 0).mean()),
                                    "strict_zero_fraction": float((values == 0).mean()),
                                    "zero_mean_fraction": float((data == 0).mean()),
                                    "allow_dropped": allow,
                                    "undefined_reason": "empty_in_range_histogram" if not counts.sum() else ""})
                    for j, count in enumerate(counts):
                        histograms.append({**meta, "bin_index": j, "left": bins.edges[j], "right": bins.edges[j+1],
                                           "count": int(count), "n_values": len(data), "dropped_mass": dropped})
    return pd.DataFrame(metrics), pd.DataFrame(histograms)


def common_endpoint_bins(runs):
    """Cover all endpoint feature means at the existing widths, recorded separately."""
    result = {}
    for axis, reduce_axis in (("population", 1), ("lifetime", 0)):
        widths = {c["analysis"][f"{axis}_bins"]["width"] for c, _, _ in runs.values()}
        if len(widths) != 1:
            raise ValueError("Common range requires a shared bin width")
        width = widths.pop()
        high = max(
            float(a["features"][f"h_{c['analysis']['split']}"].mean(axis=reduce_axis).max())
            for c, a, _ in runs.values()
        )
        result[axis] = BinSpec(0, max(2, int(np.ceil(high / width))) * width, width)
    return result


def endpoint_anchors(runs, activity):
    """RF-08: compare current saved entropy/silhouette with their source features."""
    rows = []
    for path, (c, a, t) in runs.items():
        identity = _identity(path, c)
        endpoint = activity[
            (activity.run == path) & (activity.source == "features") & (activity.bin_scope == "main")
        ]
        for e in t["entropy"].itertuples():
            values = endpoint[endpoint.axis == e.axis].iloc[0]
            for metric in ("H", "Hhat", "S", "S_prime", "zero_bin_frac", "dropped_mass"):
                saved, actual = getattr(e, metric), values[metric]
                rows.append({**identity, "metric": f"{e.axis}.{metric}", "saved": saved, "recomputed": actual,
                             "equal": bool(np.isclose(saved, actual, equal_nan=True)),
                             "undefined": not np.isfinite(actual)})

        split = c["analysis"]["split"]
        score = representation_silhouette(
            a["features"][f"h_{split}"].T, a["features"][f"y_{split}"],
            metric=c["analysis"]["metric"], center=c["analysis"]["center"]
        )
        score = np.nan if score is None else score
        saved = t["silhouette"].silhouette.iloc[0]
        rows.append({**identity, "metric": "silhouette", "saved": saved, "recomputed": score,
                     "equal": bool(np.isclose(saved, score, equal_nan=True)), "undefined": not np.isfinite(score)})
    return pd.DataFrame(rows)


def analysis_target_tables(summary, targets=None):
    """Return rule-based candidates and effective per-model display targets.

    Only the final classifier validation loss determines automatic selection.
    Explicit targets change the display, not the recorded candidate evidence.
    """
    trained = summary[summary.arm.str.endswith("_trained")]
    candidates = classifier_target_candidates(trained)
    automatic = targets is None
    if automatic:
        targets = dict(zip(candidates.model_code, candidates.candidate_targ))
    if set(targets) != set(candidates.model_code):
        raise ValueError("Explicit display targets required for every trained model")
    for model, targ in targets.items():
        if trained[(trained.model_code == model) & (trained.targ == targ)].empty:
            raise ValueError("Display target is absent")

    effective = pd.DataFrame([{
        "model_code": model, "targ": targ,
        "method": TARGET_CRITERION if automatic else "explicit_display_targets",
        "median_final_val_loss": trained.loc[
            (trained.model_code == model) & (trained.targ == targ), "final_val_loss"
        ].median(),
        "tiebreak": TIEBREAK_RULE if automatic else "not_applicable"}
        for model, targ in targets.items()])
    return candidates, effective


def check_figure_inputs(runs, expected, selection_directory, targets):
    """Check saved populations/HP conditions; return HP report and coverage.

    Coverage carries the source hashes and effective targets needed at save
    time. No figure-specific aggregation, plotting or state inference occurs.
    """
    require_complete(expected)
    if expected.attrs.get("readout_only"):
        raise ValueError("Full figure inputs unavailable from a readout-only load")
    if set(runs) != set(expected.run):
        raise ValueError("Loaded runs differ from the planned population")
    _display_population(expected, targets)

    selection_path = Path(selection_directory).resolve()
    selection_hashes = _source_hashes(selection_path)
    report, hp_specs = read_selection(selection_path)
    if report["sweep_config"]["monitor_probe_split"] != "val":
        raise ValueError("Target candidate evidence requires validation monitor values (monitor_probe_split='val')")
    sweep_config = experiment_config_from_json(json.dumps(report["sweep_config"]))

    # The figure population must cover every selected HP cell and matched baseline seed.
    hp_seeds = {s.seed_index for s in hp_specs}
    cells = {(cell["model_code"], cell["targ"]) for cell in report["cells"].values()}
    planned = expected[expected.representation == "sparse_trained"]
    wanted = {(m, t, s) for m, t in cells for s in hp_seeds}
    actual = set(planned[["model_code", "targ", "seed_index"]].itertuples(index=False, name=None))
    if actual != wanted:
        raise ValueError("Planned training population differs from all HP model/target/seed cells")
    wanted_baselines = {
        (arm, s) for arm in [*(f"{m}_initial" for m, _ in cells), "raw", "mlp_frozen"]
        for s in hp_seeds
    }
    actual_baselines = set(expected.loc[
        expected.representation != "sparse_trained", ["arm", "seed_index"]
    ].itertuples(index=False, name=None))
    if actual_baselines != wanted_baselines:
        raise ValueError("Planned baseline population does not cover all figure arms/seeds")

    for path, (c, _, _) in runs.items():
        if c["input"]["sha256"] != report["input"]["sha256"]:
            raise ValueError("HP input differs")
        _check_training_config(sweep_config, experiment_config_from_json(json.dumps(c["config"])))
        if c["config"]["n_epochs"] != report["schedule"]["n_epochs"]:
            raise ValueError("Training epochs differ from HP selection schedule")
        if c["representation"] == "sparse_trained":
            cell = report["cells"][f"{c['spec']['model_code']}@{c['spec']['targ']:g}"]
            if c["spec"]["eta"] != cell["selected_eta"]:
                raise ValueError("Used eta differs from HP selection")

    coverage = expected.copy()
    coverage.attrs["figure_sources"] = {
        "selection": {"path": str(selection_path), "sha256": selection_hashes},
        "population_sources": expected.attrs.get("population_sources", {}),
        "display_targets": dict(targets)}
    return report, coverage


def save_figure_tables(directory, tables, runs, *, targets, selection_directory, notebook_path=None,
                      coverage=None, analysis_targets=None, arrays=None, details=None):
    """Save supplied tables/arrays in a new directory after checking source hashes.

    A notebook can pass coverage from check_figure_inputs and its effective
    analysis_targets separately, without exporting either as an extra CSV.
    Existing all-table callers may continue to supply these inside tables.
    Return the output path; source runs and selection files are unchanged.
    """
    provenance = []
    for path, (c, _, _) in runs.items():
        hashes = _source_hashes(path)
        if hashes != c.get("loaded_source_sha256"):
            raise ValueError(f"Source changed since loading: {path}; reload before saving")
        provenance.append({"path": path, "sha256": hashes})

    selection = Path(selection_directory).resolve()
    coverage = tables["coverage"] if coverage is None else coverage
    analysis_targets = tables["analysis_targets"] if analysis_targets is None else analysis_targets
    sources = coverage.attrs["figure_sources"]
    if sources["display_targets"] != targets:
        raise ValueError("Display targets changed since aggregation; rebuild tables before saving")
    if (
        not analysis_targets.model_code.is_unique
        or analysis_targets.set_index("model_code").targ.to_dict() != targets
    ):
        raise ValueError("Analysis targets differ from the recorded display targets")
    if sources["selection"] != {"path": str(selection), "sha256": _source_hashes(selection)}:
        raise ValueError("HP selection changed since aggregation; reload before saving")
    for path, checksum in sources["population_sources"].items():
        if not Path(path).is_file() or hashlib.sha256(Path(path).read_bytes()).hexdigest() != checksum:
            raise ValueError("Expected population changed since loading; reload before saving")

    write_result(
        directory, {
            "representation": "figure_tables", "source": provenance,
            "selection": sources["selection"], "population_sources": sources["population_sources"],
            "display_targets": targets, "analysis_targets": analysis_targets.to_dict("records"),
            "target_criterion": TARGET_CRITERION, "sd_ddof": 0,
            "quantile_interpolation": "linear", "thresholds": list(THRESHOLDS),
            "deferred": ["AD-01 inference", "RF-01 fixed-point analysis", "SP-02/03/06 UMAP"],
            **({"additional_analysis": details, "deferred": []} if details is not None else {}),
        },
        tables=tables, arrays=arrays, script_path=notebook_path,
    )
    return Path(directory)
