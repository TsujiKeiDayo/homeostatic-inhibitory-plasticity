"""Run sparse-network and baseline experiments with a linear classifier.

Input images and readout features have samples in rows. The dynamical system
has samples in columns. Results are returned in memory for separate saving.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .adapt import AdaptationResult, run_adaptation
from .config import ExperimentConfig, RunSpec
from .data import Dataset, load_dataset
from .encoders import labels_dict, sparse_features, raw_features, mlp_frozen_features
from .metrics.bins import BinSpec, LIFETIME_BINS, POPULATION_BINS
from .metrics.entropy import scores_from_counts
from .metrics.silhouette import representation_silhouette
from .readout import ReadoutResult, train_readout
from .rng import STREAMS, root_seed, stream, stream_torch_seed


@dataclass(frozen=True)
class AnalysisSettings:
    """Definitions of the existing entropy and silhouette measurements."""

    split: str = "test"
    population_bins: BinSpec = POPULATION_BINS
    lifetime_bins: BinSpec = LIFETIME_BINS
    allow_dropped: bool = True
    metric: str = "cosine"
    center: bool = False
    metrics: tuple[str, ...] = ("entropy", "silhouette")

    def __post_init__(self):
        if self.split not in ("train", "val", "test", "test_full"):
            raise ValueError("analysis split must be train, val, test or test_full")
        if not self.metrics or set(self.metrics) - {"entropy", "silhouette"}:
            raise ValueError("analysis metrics must contain entropy and/or silhouette")


@dataclass(frozen=True)
class MlpSettings:
    """MLP training is independent of the later linear-readout training budget."""

    epochs: int = 15
    lr: float = .003
    batch_size: int = 64

    def config(self, base):
        """Copy the shared config with the MLP's training budget and learning rate."""
        if self.epochs < 1 or self.batch_size < 1 or not np.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("MLP epochs, batch_size and lr must be positive")
        return replace(base, readout_epochs=self.epochs, readout_lr=self.lr, batch_size=self.batch_size)


@dataclass
class SingleResult:
    """Features, classifier results and measurements with their recorded conditions."""

    adaptation: AdaptationResult | None
    features: dict[str, np.ndarray]  # h_* and aligned y_*; samples in rows
    readout: ReadoutResult
    entropy: pd.DataFrame
    silhouette: pd.DataFrame
    conditions: dict
    encoder_weights: dict[str, np.ndarray] | None = None


def analyse_features(features: dict[str, np.ndarray], settings: AnalysisSettings):
    """Return entropy and silhouette tables for the requested feature split.

    Features have shape (samples, units). Population rates average over units;
    lifetime rates average over samples. Disabled metrics return empty tables.
    """
    values, labels = features[f"h_{settings.split}"], features[f"y_{settings.split}"]
    validate_features(features, [settings.split])

    rows = []
    for axis, reduce_axis, bins in (
        ("population", 1, settings.population_bins),
        ("lifetime", 0, settings.lifetime_bins)
    ):
        if "entropy" not in settings.metrics:
            continue
        counts, dropped = bins.apply(
            values.mean(axis=reduce_axis), allow_dropped=settings.allow_dropped
        )
        rows.append({"split": settings.split, "axis": axis, **asdict(scores_from_counts(counts)),
                     "dropped_mass": dropped, "n_bins": bins.n_bins, **asdict(bins)})

    silhouette = pd.DataFrame(columns=["split", "silhouette", "metric", "center"])
    if "silhouette" in settings.metrics:
        score = representation_silhouette(
            values.T, labels, metric=settings.metric, center=settings.center
        )
        silhouette = pd.DataFrame([{"split": settings.split, "silhouette": score,
                                    "metric": settings.metric, "center": settings.center}])
    entropy = (
        pd.DataFrame(rows, columns=[
            "split", "axis", "H", "Hhat", "S", "zero_bin_frac", "S_prime",
            "dropped_mass", "n_bins", "lo", "hi", "width"
        ]) if not rows else pd.DataFrame(rows)
    )
    return entropy, silhouette


def _validate_data(data, config, spec, full, size="small"):
    """Validate run conditions and aligned dataset splits of the requested size."""
    for key in ("n_excitatory", "n_inhibitory", "settle_steps", "n_epochs", "batch_size",
                "readout_epochs", "readout_eval_every"):
        value = getattr(config, key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")

    if isinstance(spec.seed_index, bool) or not isinstance(spec.seed_index, int):
        raise ValueError("seed_index must be an integer index into the existing RNG tree")
    root_seed(spec.seed_index)
    if config.n_inhibitory != 1:
        raise ValueError("The current experiment supports n_inhibitory=1")
    if (
        isinstance(config.n_stabilise, bool)
        or not isinstance(config.n_stabilise, int)
        or config.n_stabilise < 0
    ):
        raise ValueError("n_stabilise must be a non-negative integer")
    for key, value in {"tau_e": config.tau_e, "tau_i": config.tau_i, "targ": spec.targ,
                       "eta": spec.eta, "readout_lr": config.readout_lr}.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
    json.dumps({"config": asdict(config), "spec": asdict(spec)}, allow_nan=False)

    # Every requested split must use the same input width and aligned labels.
    width = None
    for split in ["train", "val", "test"] + (["test_full"] if full else []):
        x_key, y_key = f"x_{split}_{size}", f"y_{split}"
        if x_key not in data or y_key not in data:
            raise ValueError(f"dataset requires {x_key} and {y_key}")
        x, y = data[x_key], data[y_key]
        if (
            x.ndim != 2 or y.ndim != 1 or len(x) != len(y)
            or not len(x) or not x.shape[1]
        ):
            raise ValueError(f"{split}: nonempty samples x inputs and aligned labels are required")
        if (
            not np.isfinite(x).all()
            or not np.issubdtype(y.dtype, np.integer)
            or (y < 0).any()
        ):
            raise ValueError(f"{split}: inputs must be finite and labels non-negative integers")
        if width is not None and x.shape[1] != width:
            raise ValueError(f"{split}: input width differs from train")
        width = x.shape[1]


def run_single(
    dataset_path: str | Path | Dataset, spec: RunSpec, config: ExperimentConfig, *,
    analysis: AnalysisSettings = AnalysisSettings(), device: str = "cpu"
) -> SingleResult:
    """Load once; initialise/adapt, freeze features, fit readout, measure.

    This function does not save results or discover/create a dataset. The input
    NPZ must explicitly contain x_{train,val,test}_small and aligned y_* arrays,
    plus test_full when requested. Original dataset metadata is recorded as
    input metadata; it does not override any supplied scientific condition.
    """
    dataset = load_dataset(dataset_path)
    data = dataset.arrays
    full = config.readout_final_eval_full_test or analysis.split == "test_full"
    _validate_data(data, config, spec, full)

    torch_device = torch.device(device)
    splits = {
        split: torch.as_tensor(
            data[f"x_{split}_small"].T, dtype=torch.float32, device=torch_device
        )
        for split in ("train", "val", "test")
    }

    # Homeostatic updates use the explicit rule, without an autograd graph.
    with torch.inference_mode():
        adapted = run_adaptation(spec, config, splits, torch_device)
    return finish_adaptation(dataset, adapted, spec, config, analysis=analysis, device=device)


def validate_features(features, splits):
    """Require finite sample-row features, aligned labels and a shared feature width."""
    width = None
    for split in splits:
        if f"h_{split}" not in features or f"y_{split}" not in features:
            raise ValueError(
                f"Saved features require h_{split} and y_{split}; retain full-test features when needed"
            )
        x, y = features[f"h_{split}"], features[f"y_{split}"]
        if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or not len(x):
            raise ValueError(f"{split}: features and labels must be aligned")
        if (
            not np.isfinite(x).all()
            or not np.issubdtype(y.dtype, np.integer)
            or (y < 0).any()
        ):
            raise ValueError(f"{split}: invalid features or labels")
        if width is not None and x.shape[1] != width:
            raise ValueError("Feature width differs across splits")
        width = x.shape[1]


def fit_readout(
    features, config, seed, *, device="cpu",
    init_stream="readout/init", shuffle_stream="readout/shuffle"
):
    """Fit a linear classifier to h_*/y_* splits and return its result and RNG seeds.

    ``seed`` is an index into the root seed tree; named streams independently
    determine classifier initialisation and batch order.
    """
    for key in ("readout_epochs", "readout_eval_every", "batch_size"):
        value = getattr(config, key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not np.isfinite(config.readout_lr) or config.readout_lr <= 0:
        raise ValueError("readout_lr must be finite and positive")
    validate_features(
        features,
        ["train", "val", "test"] + (["test_full"] if config.readout_final_eval_full_test else [])
    )

    init_seed = stream_torch_seed(seed, init_stream)
    shuffle_seed = int(stream(seed, shuffle_stream).integers(0, 2**31))
    fitted = train_readout(features, features, config, init_seed=init_seed,
                          shuffle_seed=shuffle_seed, device=torch.device(device))
    return fitted, {
        "root_seed": root_seed(seed), "readout_init": init_seed, "readout_shuffle": shuffle_seed
    }


def reconstruction_conditions(spec, config, representation):
    """Recorded reconstruction protocol; shared by saving and artifact checks."""
    numerics = {
        "dtype": "float32", "n_E": config.n_excitatory,
        "n_I": 0 if spec.model_code == "thresh" else config.n_inhibitory,
        "settling_update": "simultaneous_v_E_v_I_from_previous_state_then_o_E",
        "warm_start": "train_prefix_sequential; epoch_start_for_batches; frozen_parameters_for_probes",
        "trajectory": "initial_then_after_every_batch; C_flat_parameter_order"
    }
    if representation in ("raw", "mlp_frozen"):
        numerics = {"dtype": "float32", "encoder": representation,
                    "input_size": "big" if representation == "raw" else "small"}
    return {
        "sample_order": "zero_based_rows_within_input_split; not_original_image_ids",
        "numerics": numerics,
        "rng": {
            "root_seed": root_seed(spec.seed_index), "streams": dict(STREAMS),
            "adaptation_shuffle": "numpy.default_rng(SeedSequence(root_seed, spawn_key=(2,)))",
            "permutation": "one_stream_per_run; one_permutation_per_epoch"
        },
        "axes": {"features": "samples_features", "activity": "units_samples"}
    }


def scientific_conditions(dataset, spec, config, analysis, device, representation="sparse_trained"):
    """Collect effective conditions and input identity for saving and reuse checks."""
    return {
        "spec": asdict(spec), "config": asdict(config), "analysis": asdict(analysis),
        "input": dataset.source, "representation": representation,
        "device": str(torch.device(device)),
        "output_version": 1, **reconstruction_conditions(spec, config, representation),
        "readout_streams": {"init": "readout/init", "shuffle": "readout/shuffle"}
    }


def finish_adaptation(dataset, adapted, spec, config, *, analysis=AnalysisSettings(), device="cpu"):
    """Extract frozen features, train their classifier and return measured results."""
    full = config.readout_final_eval_full_test or analysis.split == "test_full"
    with torch.inference_mode():
        features = sparse_features(
            adapted.params_trained, dataset.arrays, config, torch.device(device),
            include_full_test=full
        )
    features.update(labels_dict(dataset.arrays, include_full_test=full))

    # Only the readout is trained by autograd. Keep it outside inference_mode.
    fitted, seeds = fit_readout(features, config, spec.seed_index, device=device)
    entropy, silhouette = analyse_features(features, analysis)
    conditions = {**scientific_conditions(dataset, spec, config, analysis, device), "seeds": seeds}
    return SingleResult(adapted, features, fitted, entropy, silhouette, conditions)


def _baseline_conditions(
    dataset, spec, config, analysis, device, representation, mlp, init_stream=None
):
    """The same baseline identity is used before execution and when saving."""
    conditions = scientific_conditions(dataset, spec, config, analysis, device, representation)
    conditions["spec"] = {
        "seed_index": spec.seed_index,
        **({"model_code": spec.model_code} if representation == "sparse_initial" else {})
    }
    init_stream = init_stream or (
        "mlp/readout_init" if representation == "mlp_frozen" else "readout/init"
    )
    conditions["readout_streams"] = {"init": init_stream, "shuffle": "readout/shuffle"}
    if representation == "mlp_frozen":
        conditions["mlp"] = asdict(mlp)
    return conditions


def run_representation(
    dataset_path, config, seed, *, representation, model_code="rec",
    mlp=MlpSettings(), analysis=AnalysisSettings(), device="cpu", init_stream=None
):
    """Build baseline features, train their classifier and return an unsaved SingleResult.

    ``representation`` selects sparse_initial, raw or mlp_frozen. Only the MLP
    encoder is trained; all three then use a separate linear classifier.
    ``seed`` is an index into the root seed tree, shared by the named streams.
    """
    from .model.init import build_params
    from .mlp import train_mlp_e2e

    dataset = load_dataset(dataset_path)
    full = config.readout_final_eval_full_test or analysis.split == "test_full"
    # Baselines use model/seed only; dummy target/eta values are not saved.
    spec = RunSpec(model_code, 1., 1., seed)
    _validate_data(
        dataset.arrays, config, spec, full, size="big" if representation == "raw" else "small"
    )

    weights = None
    if representation == "sparse_initial":
        params = build_params(
            model_code, config, seed, dataset.arrays["x_train_small"].shape[1],
            torch.device(device)
        )
        weights = params.to_arrays()
        with torch.inference_mode():
            features = sparse_features(
                params, dataset.arrays, config, torch.device(device), include_full_test=full
            )
    elif representation == "raw":
        features = raw_features(dataset.arrays, include_full_test=full)
    elif representation == "mlp_frozen":
        model = train_mlp_e2e(
            dataset.arrays["x_train_small"], dataset.arrays["y_train"], mlp.config(config),
            init_seed=stream_torch_seed(seed, "mlp/init"),
            shuffle_seed=int(stream(seed, "mlp/shuffle").integers(0, 2**31)),
            device=torch.device(device)
        )
        weights = {
            key: value.detach().cpu().numpy().copy()
            for key, value in model.state_dict().items()
        }
        with torch.no_grad():
            features = mlp_frozen_features(
                model, dataset.arrays, torch.device(device), include_full_test=full
            )
    else:
        raise ValueError("representation must be sparse_initial, raw or mlp_frozen")

    features.update(labels_dict(dataset.arrays, include_full_test=full))
    conditions = _baseline_conditions(
        dataset, spec, config, analysis, device, representation, mlp, init_stream
    )
    fitted, seeds = fit_readout(features, config, seed, device=device,
                               init_stream=conditions["readout_streams"]["init"])
    entropy, silhouette = analyse_features(features, analysis)
    conditions["seeds"] = seeds
    if representation == "mlp_frozen":
        seeds.update(mlp_init=stream_torch_seed(seed, "mlp/init"),
                     mlp_shuffle=int(stream(seed, "mlp/shuffle").integers(0, 2**31)))
    return SingleResult(None, features, fitted, entropy, silhouette, conditions, weights)
