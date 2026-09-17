"""Batched adaptation: R runs of one variant stacked along a run axis (SPEC V12).

``run_adaptation_batch`` follows the same protocol as ``adapt.run_adaptation``
for each RunSpec — the same initialisation streams, the same per-seed batch
order, the same warm starts and probes — but settles all R networks in one
tensor program to share kernel launches across runs.

Numerics: batched matmul rounds differently from single matmul, so results
need not be bit-identical to the sequential path. Tolerances depend on the
quantity and condition; see tests/dynamics/test_engine_equivalence.py and
history/migration/M5_REVIEW.md.
Production runs record the engine and actual batch width in conditions.json.
"""

from __future__ import annotations

import numpy as np
import torch

from ..adapt import AdaptationResult, _adaptive_snapshot, monitor_stats, probe_summaries
from ..config import ExperimentConfig, RunSpec
from ..model.init import build_params
from ..model.params import ModelParams
from ..model.plasticity import eta_schedule, update
from ..rng import stream
from ..simulate import encode, warm_start_state


def run_adaptation_batch(
    specs: list[RunSpec],
    config: ExperimentConfig,
    splits: dict[str, torch.Tensor],
    device: torch.device,
    *,
    keep_trajectory: bool = True,
    analysis_probes: bool = True,
) -> list[AdaptationResult]:
    """Adapt every spec (all of one ``model_code``) under one ``config``.

    ``splits`` maps train/val/test to (nIn, n_samples) tensors shared across runs.
    Returns one ``AdaptationResult`` per spec, in order. ``keep_trajectory=False``
    leaves ``param_trajectory`` and ``trajectory_index`` empty; ``params_initial``
    is still retained. ``analysis_probes=False`` skips first/final-epoch analysis
    probes. Both flags leave updates and per-epoch monitor/train measurements unchanged.
    """
    if not specs:
        return []
    codes = {spec.model_code for spec in specs}
    if len(codes) != 1:
        raise ValueError(f"one batch holds one variant; got {sorted(codes)}")
    model_code = specs[0].model_code
    for spec in specs:
        spec.validate_against(config)   # warns, never raises (V7)

    x_train = splits["train"]
    x_monitor = splits[config.monitor_probe_split]
    x_analysis = splits[config.analysis_probe_split]
    n_input, n_samples = int(x_train.shape[0]), int(x_train.shape[1])

    # Per-run initialisation through the reference path, then stacked: the
    # starting weights are bit-identical to a sequential run's.
    singles = [build_params(model_code, config, spec.seed_index, n_input, device)
               for spec in specs]
    params = ModelParams.stack(singles)
    n_runs = len(specs)
    targ = torch.tensor([spec.targ for spec in specs], dtype=torch.float32, device=device)

    results = [AdaptationResult(params_initial=p.clone(), params_trained=p) for p in singles]
    if keep_trajectory:
        for result, single in zip(results, singles):
            result.param_trajectory.append(_adaptive_snapshot(single))
            result.trajectory_index.append((-1, -1))

    # One shuffle stream per run, created once, exactly as the sequential path does.
    shuffle_rngs = [stream(spec.seed_index, "shuffle") for spec in specs]
    analysis_epochs = {0, config.n_epochs - 1}

    for epoch in range(config.n_epochs):
        # Per-run eta this epoch, kept as Python floats for the records.
        etas = [eta_schedule(spec.eta, epoch, config.eta_decay, config.eta_decay_start_epoch)
                for spec in specs]
        eta_vec = torch.tensor(etas, dtype=torch.float32, device=device)
        # Shared train samples, per-run weights -> (R, nE, 1) warm state.
        warm = warm_start_state(params, x_train, config)
        perms = [rng.permutation(n_samples) for rng in shuffle_rngs]

        # Settle each run's shuffled batches, then apply homeostatic updates.
        for b_start in range(0, n_samples, config.batch_size):
            idx = np.stack([perm[b_start:b_start + config.batch_size] for perm in perms])
            idx_t = torch.as_tensor(idx, device=device)               # (R, b)
            # x_train[:, idx] is (nIn, R, b); the run axis must lead.
            x_batch = x_train[:, idx_t].permute(1, 0, 2).contiguous()  # (R, nIn, b)
            state = encode(params, x_batch, config, warm)
            pre = state.v_I if model_code in ("rec", "ff") else None
            params = update(params, state.o_E, pre, targ=targ, eta=eta_vec)
            if keep_trajectory:
                # One host copy per update for all runs, then split per run.
                adapted = params.w_EI if model_code in ("rec", "ff") else params.theta
                snapshot = adapted.detach().cpu().numpy().reshape(n_runs, -1)
                for r, result in enumerate(results):
                    result.param_trajectory.append(snapshot[r].copy())
                    result.trajectory_index.append((epoch, int(b_start)))

        # Measure with frozen parameters and a warm state from the train split.
        warm_probe = warm_start_state(params, x_train, config)
        monitor_state = encode(params, x_monitor, config, warm_probe)
        stats = monitor_stats(monitor_state.o_E, monitor_state.v_I, targ,
                              config.monitor_var_weight)
        stats_np = {key: value.detach().cpu().numpy() for key, value in stats.items()}

        # Fixed-point rates refer to train data, separate from the monitor split.
        train_state = encode(params, x_train, config, warm_probe)
        stats_train = monitor_stats(train_state.o_E, train_state.v_I, targ,
                                    config.monitor_var_weight)
        train_np = {key: stats_train[key].detach().cpu().numpy()
                    for key in ("rate_mean", "rate_wmean")}

        for r, result in enumerate(results):
            l_mean, rate_var = float(stats_np["l_mean"][r]), float(stats_np["rate_var"][r])
            result.monitor_records.append({
                "l_mean": l_mean,
                "rate_var": rate_var,
                # Recombined in float64 from the components, like monitor_loss.
                "composite": l_mean + config.monitor_var_weight * rate_var,
                "rate_mean": float(stats_np["rate_mean"][r]),
                "rate_wmean": float(stats_np["rate_wmean"][r]),
                "rate_mean_train": float(train_np["rate_mean"][r]),
                "rate_wmean_train": float(train_np["rate_wmean"][r]),
                "epoch": epoch,
                "eta": etas[r],
            })

        if analysis_probes and epoch in analysis_epochs:
            analysis_state = encode(params, x_analysis, config, warm_probe)
            for r, result in enumerate(results):
                single = params.select(r)
                result.analysis_probes[epoch] = probe_summaries(
                    single, analysis_state.o_E[r], analysis_state.v_I[r], x_analysis)
                result.analysis_activity[epoch] = (
                    analysis_state.o_E[r].detach().cpu().numpy().astype(np.float32))

    for r, result in enumerate(results):
        result.params_trained = params.select(r)
    return results


def chunk(items: list, size: int) -> list[list]:
    """Consecutive chunks of at most ``size`` items."""
    if size < 1:
        raise ValueError("chunk size must be >= 1")
    return [items[i:i + size] for i in range(0, len(items), size)]
