"""Homeostatic adaptation loop (C3/C7; V6).

One run = one (model, targ, eta, seed). The monitor probe records rate
statistics after every epoch on the monitor split (val by default). The
analysis probe stores per-axis summaries and raw activity after the first
and final epochs on the analysis split (test by default).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .config import ExperimentConfig, RunSpec
from .metrics.bins import LIFETIME_BINS, POPULATION_BINS
from .model.dynamics import input_drive
from .model.init import build_params
from .model.params import ModelParams
from .model.plasticity import eta_schedule, update
from .rng import stream
from .simulate import encode, warm_start_state


def monitor_stats(
    o_E: torch.Tensor,
    v_I: torch.Tensor | None,
    targ: float | torch.Tensor,
    var_weight: float,
) -> dict[str, torch.Tensor]:
    """Compute rate statistics for ``o_E`` of shape (..., nE, n_samples).

    A leading run axis (stacked networks, SPEC V12) is carried through every
    output. ``v_I`` has shape (..., nI, n_samples), and ``targ`` is scalar or
    one value per run.

    Returned tensors have the leading shape ``(...)``:

      l_mean      (mean - targ)^2, the population-mean error
      rate_var    variance of the per-unit mean rates ACROSS units (population
                  variance, ``unbiased=False``), so l_mean + rate_var is exactly
                  the per-unit MSE mean_i (r_i - targ)^2
      composite   l_mean + var_weight * rate_var, the HP eta-selection objective
                  (SPEC V10: var_weight = 1 makes it the per-unit MSE)
      rate_mean   plain mean rate
      rate_wmean  v_I-weighted mean rate, sum_b o_ib v_Ib / sum_b v_Ib averaged
                  over units: the quantity the W_EI rule actually drives to
                  targ, i.e. its fixed point (SPEC §2; CL9). Equals rate_mean
                  when ``v_I`` is absent (thresh) or identically zero.

    Used for measurement, not as a differentiated training objective.
    """
    # (..., nE): each unit's mean rate over the whole split.
    unit_means = o_E.mean(dim=-1)
    rate_mean = unit_means.mean(dim=-1)
    targ_t = torch.as_tensor(targ, dtype=o_E.dtype, device=o_E.device)
    l_mean = (rate_mean - targ_t) ** 2
    # unbiased=False: the variance of the units we have, not an estimate.
    rate_var = unit_means.var(dim=-1, unbiased=False)

    if v_I is not None and v_I.shape[-2] > 0:
        # (..., 1, n_samples) presynaptic mass per sample. Zero total mass
        # (silent inhibition) falls back to uniform weights, i.e. the plain mean.
        weight = v_I.sum(dim=-2, keepdim=True)
        total = weight.sum(dim=-1, keepdim=True)
        uniform = torch.full_like(weight, 1.0 / weight.shape[-1])
        weight = torch.where(total > 0, weight / total.clamp_min(1e-30), uniform)
        rate_wmean = (o_E * weight).sum(dim=-1).mean(dim=-1)
    else:
        rate_wmean = rate_mean

    return {
        "l_mean": l_mean,
        "rate_var": rate_var,
        "composite": l_mean + var_weight * rate_var,
        "rate_mean": rate_mean,
        "rate_wmean": rate_wmean,
    }


def monitor_loss(
    o_E: torch.Tensor,
    targ: float,
    var_weight: float,
    v_I: torch.Tensor | None = None,
) -> dict[str, float]:
    """``monitor_stats`` for one network, as plain floats (one per-epoch record).

    ``composite`` is recombined in float64 from the float components, matching
    the record written before ``rate_wmean`` existed.
    """
    stats = monitor_stats(o_E, v_I, targ, var_weight)
    l_mean, rate_var = float(stats["l_mean"]), float(stats["rate_var"])
    return {
        "l_mean": l_mean,
        "rate_var": rate_var,
        "composite": l_mean + var_weight * rate_var,
        "rate_mean": float(stats["rate_mean"]),
        "rate_wmean": float(stats["rate_wmean"]),
    }


def train_probe(
    o_E: torch.Tensor,
    v_I: torch.Tensor | None,
    targ: float,
    var_weight: float,
) -> dict[str, float]:
    """The two fixed-point rates measured on the TRAINING split, for one network.

    The W_EI / theta rules drive the rate on the data they see; the monitor
    split (val) adds a generalisation gap on top (SPEC §2, CL9). Recorded every
    epoch beside the monitor record; a measurement only, never used to select.
    """
    stats = monitor_stats(o_E, v_I, targ, var_weight)
    return {
        "rate_mean_train": float(stats["rate_mean"]),
        "rate_wmean_train": float(stats["rate_wmean"]),
    }


def probe_summaries(
    params: ModelParams,
    o_E: torch.Tensor,
    v_I: torch.Tensor,
    x: torch.Tensor,
) -> dict[str, np.ndarray]:
    """Per-axis summary vectors for one analysis probe (design §6 summaries schema).

    Reduces the (nE, n_samples) activity along two axes: ``_pop`` keys reduce
    over UNITS (one value per sample), ``_life`` keys over SAMPLES (one per unit).
    """
    o = o_E.detach().cpu().numpy()
    r_pop = o.mean(axis=0)          # (n_samples,)
    r_life = o.mean(axis=1)         # (nE,)

    # V4: retain out-of-range mass so histogram truncation remains visible.
    hist_pop, dropped_pop = POPULATION_BINS.apply(r_pop, allow_dropped=True)
    hist_life, dropped_life = LIFETIME_BINS.apply(r_life, allow_dropped=True)

    # Currents into each E unit, for E/I balance; both are positive magnitudes.
    e_input = input_drive(params, x)                      # (nE, B)
    if params.model_code in ("rec", "ff"):
        i_input = params.w_EI @ v_I                       # (nE, B)
    else:
        # thresh has no inhibitory pathway; zeros keep the array set uniform.
        i_input = torch.zeros_like(e_input)

    return {
        "r_pop": r_pop.astype(np.float32),
        "r_life": r_life.astype(np.float32),
        # Participation only, ignoring magnitude: fraction of entries above zero.
        "active_frac_pop": (o > 0).mean(axis=0).astype(np.float32),
        "active_frac_life": (o > 0).mean(axis=1).astype(np.float32),
        # Raw norms: sparseness indices are L1/L2 ratios, formed by later stages.
        "l1_pop": np.abs(o).sum(axis=0).astype(np.float32),
        "l2_pop": np.sqrt((o**2).sum(axis=0)).astype(np.float32),
        "l1_life": np.abs(o).sum(axis=1).astype(np.float32),
        "l2_life": np.sqrt((o**2).sum(axis=1)).astype(np.float32),
        "hist_pop": hist_pop.astype(np.int64),
        "hist_life": hist_life.astype(np.int64),
        # [population, lifetime] mass that fell outside the histogram range.
        "dropped_mass": np.array([dropped_pop, dropped_life], dtype=np.float32),
        # (nE,) mean excitatory and inhibitory current per unit.
        "e_input_life": e_input.mean(dim=1).detach().cpu().numpy().astype(np.float32),
        "i_input_life": i_input.mean(dim=1).detach().cpu().numpy().astype(np.float32),
    }


@dataclass
class AdaptationResult:
    """Everything one run produces, before it is serialised to disk.

    ``param_trajectory`` snapshots the adapting parameter after EVERY batch
    update, not every epoch; the matching ``trajectory_index`` entry is
    (epoch, batch_start), with (-1, -1) for the untrained starting point.
    """

    params_initial: ModelParams
    params_trained: ModelParams
    monitor_records: list[dict] = field(default_factory=list)  # per epoch
    param_trajectory: list[np.ndarray] = field(default_factory=list)
    trajectory_index: list[tuple[int, int]] = field(default_factory=list)
    analysis_probes: dict[int, dict[str, np.ndarray]] = field(default_factory=dict)
    analysis_activity: dict[int, np.ndarray] = field(default_factory=dict)  # epoch -> o_E

    def summaries_arrays(self) -> dict[str, np.ndarray]:
        """Collect monitor traces, parameter trajectories and probes for NPZ output.

        The key names are part of the artifact schema: ``monitor_*`` are
        parallel per-epoch traces aligned by index, probe vectors are prefixed
        ``probe_ep<NN>_``.

        Requires a nonempty parameter trajectory; batched HP results produced
        with ``keep_trajectory=False`` cannot be exported through this method.
        """
        arrays: dict[str, np.ndarray] = {
            "monitor_epoch": np.array([r["epoch"] for r in self.monitor_records], dtype=np.int64),
            "monitor_composite": np.array(
                [r["composite"] for r in self.monitor_records], dtype=np.float64
            ),
            "monitor_l_mean": np.array([r["l_mean"] for r in self.monitor_records], dtype=np.float64),
            "monitor_rate_var": np.array([r["rate_var"] for r in self.monitor_records], dtype=np.float64),
            "monitor_rate_mean": np.array(
                [r["rate_mean"] for r in self.monitor_records], dtype=np.float64
            ),
            # v_I-weighted mean rate: the W_EI rule's fixed point (CL9; A7).
            "monitor_rate_wmean": np.array(
                [r["rate_wmean"] for r in self.monitor_records], dtype=np.float64
            ),
            # The same two rates on the training split, where the fixed point
            # actually holds (CL9; A7 reads the final value).
            "monitor_rate_mean_train": np.array(
                [r["rate_mean_train"] for r in self.monitor_records], dtype=np.float64
            ),
            "monitor_rate_wmean_train": np.array(
                [r["rate_wmean_train"] for r in self.monitor_records], dtype=np.float64
            ),
            "monitor_eta": np.array([r["eta"] for r in self.monitor_records], dtype=np.float64),
            "param_trajectory": np.stack(self.param_trajectory).astype(np.float32),
            "trajectory_epoch": np.array([e for e, _ in self.trajectory_index], dtype=np.int64),
            "trajectory_batch_start": np.array(
                [b for _, b in self.trajectory_index], dtype=np.int64
            ),
        }

        # Zero-padded epoch keeps the keys sorting in temporal order.
        for epoch, vectors in self.analysis_probes.items():
            for key, value in vectors.items():
                arrays[f"probe_ep{epoch:02d}_{key}"] = value
        return arrays


def _adaptive_snapshot(params: ModelParams) -> np.ndarray:
    """Return the adapting parameter, W_EI or theta, as a flat host array."""
    if params.model_code in ("rec", "ff"):
        return params.w_EI.detach().cpu().numpy().ravel()
    return params.theta.detach().cpu().numpy().ravel()


def run_adaptation(
    spec: RunSpec,
    config: ExperimentConfig,
    splits: dict[str, torch.Tensor],
    device: torch.device,
) -> AdaptationResult:
    """Execute one run: initialise, adapt for ``config.n_epochs``, probe, return.

    ``splits`` maps train/val/test to (nIn, n_samples) tensors. Analysis
    probes are measured after epochs 0 and ``n_epochs - 1``; only the initial
    parameter snapshot represents the untrained state.
    """
    # V7 allows targets at or below the initial threshold, with a warning.
    spec.validate_against(config)

    x_train = splits["train"]
    x_monitor = splits[config.monitor_probe_split]
    x_analysis = splits[config.analysis_probe_split]

    params = build_params(spec.model_code, config, spec.seed_index, x_train.shape[0], device)
    result = AdaptationResult(params_initial=params.clone(), params_trained=params)
    # Snapshot 0 is the untrained state; (-1, -1) marks "before any update".
    result.param_trajectory.append(_adaptive_snapshot(params))
    result.trajectory_index.append((-1, -1))

    # Created once, so epochs consume a fixed, reproducible sequence of shuffles.
    shuffle_rng = stream(spec.seed_index, "shuffle")
    n_samples = x_train.shape[1]
    # Only the first and last epoch pay for the expensive analysis probe (V6).
    analysis_epochs = {0, config.n_epochs - 1}

    for epoch in range(config.n_epochs):
        # Per-epoch setup: decayed eta, one warm start shared by all batches, reshuffle.
        eta = eta_schedule(spec.eta, epoch, config.eta_decay, config.eta_decay_start_epoch)
        warm = warm_start_state(params, x_train, config)
        perm = shuffle_rng.permutation(n_samples)

        # Settle each batch before applying its homeostatic update.
        for b_start in range(0, n_samples, config.batch_size):
            # A short trailing batch is harmless: the update divides by the real width.
            batch_idx = perm[b_start : b_start + config.batch_size]
            x_batch = x_train[:, batch_idx]
            state = encode(params, x_batch, config, warm)
            # Presynaptic factor is the settled v_I — hence one update per batch.
            pre = state.v_I if spec.model_code in ("rec", "ff") else None
            # params is REBOUND, not mutated, so each snapshot stays a true record.
            params = update(params, state.o_E, pre, targ=spec.targ, eta=eta)
            result.param_trajectory.append(_adaptive_snapshot(params))
            result.trajectory_index.append((epoch, int(b_start)))

        # Measure with frozen weights and a shared train-derived warm start (V3).
        warm_probe = warm_start_state(params, x_train, config)
        monitor_state = encode(params, x_monitor, config, warm_probe)
        record = monitor_loss(
            monitor_state.o_E, spec.targ, config.monitor_var_weight,
            v_I=monitor_state.v_I,
        )

        # The fixed point is a condition on the TRAINING data (SPEC §2): probe
        # that split too, so the residual can be reported without the held-out
        # generalisation gap (CL9; A7). Measurement only; weights stay frozen.
        train_state = encode(params, x_train, config, warm_probe)
        record.update(train_probe(
            train_state.o_E, train_state.v_I, spec.targ, config.monitor_var_weight
        ))
        record.update(epoch=epoch, eta=eta)
        result.monitor_records.append(record)

        if epoch in analysis_epochs:
            # Epoch 0 is after the first training pass, not an untrained baseline.
            analysis_state = encode(params, x_analysis, config, warm_probe)
            result.analysis_probes[epoch] = probe_summaries(
                params, analysis_state.o_E, analysis_state.v_I, x_analysis
            )
            # Keep raw (nE, n_analysis) activity for downstream feature analyses.
            result.analysis_activity[epoch] = (
                analysis_state.o_E.detach().cpu().numpy().astype(np.float32)
            )

    result.params_trained = params
    return result
