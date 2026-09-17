"""Explicit extra inference from saved parameters; never perform learning updates.

AD-01 follows the original batch order and restores each pre-update parameter.
RF-01 measures the final saved model using its recorded split and settling loop.
Figure-oriented grouping and plotting belong in the notebooks.
"""
import json

import numpy as np
import pandas as pd
import torch

from .config import experiment_config_from_json
from .model.params import ModelParams
from .model.dynamics import Decays, input_drive, step
from .rng import stream
from .simulate import encode, warm_start_state


def _inputs(conditions, dataset):
    """Check input identity and restore process-wide Torch numeric settings.

    Return config, recorded device and train inputs shaped (input units, samples).
    Thread count, matmul precision and TF32 settings remain changed after return.
    """
    if dataset.source["sha256"] != conditions["input"]["sha256"]:
        raise ValueError("Inference input differs from the saved experiment")
    config = experiment_config_from_json(json.dumps(conditions["config"]))
    device = torch.device(conditions["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("The recorded CUDA device is unavailable; no CPU fallback")

    environment = conditions["environment"]
    torch.set_num_threads(environment["torch_threads"])
    torch.set_float32_matmul_precision(environment["float32_matmul_precision"])
    torch.backends.cuda.matmul.allow_tf32 = environment["cuda_matmul_allow_tf32"]
    torch.backends.cudnn.allow_tf32 = environment["cudnn_allow_tf32"]
    train = torch.as_tensor(dataset.arrays["x_train_small"].T, device=device)
    return config, device, train


def _restore(conditions, arrays, device, index):
    """Restore an adaptive snapshot without changing any non-adaptive weight."""
    model = conditions["spec"]["model_code"]
    params = ModelParams.from_arrays(model, arrays["weights_initial"], device)
    values = torch.as_tensor(arrays["summaries"]["param_trajectory"][index], device=device)
    return (
        params.with_(theta=values.reshape(params.theta.shape))
        if model == "thresh" else params.with_(w_EI=values.reshape(params.w_EI.shape))
    )


def _probe_check(actual, saved, *, stage):
    """Record floating-point agreement, using NumPy's established allclose tolerances."""
    actual = actual.detach().cpu().numpy()
    equal = bool(np.allclose(actual, saved, rtol=1e-5, atol=1e-8))
    return {"stage": stage, "max_abs_error": float(np.max(np.abs(actual-saved))),
            "rtol": 1e-5, "atol": 1e-8, "equal": equal}


def training_order_activity(conditions, arrays, dataset):
    """Return per-sample population activity in training order and endpoint checks.

    The warm state is rebuilt once at the beginning of each epoch. Each batch
    restores the saved parameter BEFORE that batch's update. No update function
    or initializer is called. Sample indices identify rows within the input split.
    The two returned tables contain sample activity and saved-probe agreement.
    Restores the recorded process-wide Torch numeric settings without resetting them.
    """
    config, device, train = _inputs(conditions, dataset)
    q = arrays["summaries"]
    n = train.shape[1]
    order = stream(conditions["spec"]["seed_index"], "shuffle")
    parts, checks, index = [], [], 0

    with torch.inference_mode():
        for epoch in range(config.n_epochs):
            params = _restore(conditions, arrays, device, index)
            warm = warm_start_state(params, train, config)
            permutation = order.permutation(n)
            for start in range(0, n, config.batch_size):
                if (
                    q["trajectory_epoch"][index+1], q["trajectory_batch_start"][index+1]
                ) != (epoch, start):
                    raise ValueError("Saved trajectory does not match training batch order")
                params = _restore(conditions, arrays, device, index)
                sample = permutation[start:start+config.batch_size]
                state = encode(params, train[:, sample], config, warm)
                parts.append(pd.DataFrame({
                    "epoch": epoch, "batch_start": start,
                    "parameter_index": index, "order_index": epoch*n+np.arange(start, start+len(sample)),
                    "sample_index": sample, "label": dataset.arrays["y_train"][sample],
                    "population_activity": state.o_E.mean(dim=0).cpu().numpy(),
                }))
                index += 1

            # Probe checkpoints use the post-epoch parameters, after the last update.
            if epoch in (0, config.n_epochs-1):
                params = _restore(conditions, arrays, device, index)
                warm = warm_start_state(params, train, config)
                split = config.analysis_probe_split
                x = torch.as_tensor(dataset.arrays[f"x_{split}_small"].T, device=device)
                probe = encode(params, x, config, warm)
                checks.append(_probe_check(
                    probe.o_E, arrays["activity"][f"o_E_ep{epoch:02d}"], stage=f"ep{epoch:02d}",
                ))

    if index != len(q["param_trajectory"])-1:
        raise ValueError("Unused or missing saved trajectory rows")
    return pd.concat(parts, ignore_index=True), pd.DataFrame(checks)


def local_modes(params, state, decays, *, tolerance=1e-4):
    """Local discrete-map modes for the recorded n_I=1 model, per sample.

    rec has n_E-1 copies of dec_E and the two roots of
    z^2-(dec_E+dec_I)z+dec_E*dec_I+k, where k sums w_IE*w_EI over
    excitatory units active for that sample. ff has dec_E and dec_I;
    thresh only dec_E. log(tol)/log(radius) is a local mode-decay
    estimate, not a bound on nonlinear convergence time.
    Return one row per sample; state columns are samples. Undefined estimates
    remain NaN with a reason, including samples on a ReLU boundary.
    """
    if params.n_inhibitory not in (0, 1):
        raise ValueError("This adopted local formula requires n_I=1 (or thresh)")

    margin = (state.v_E-params.theta[:, None]).detach().cpu().numpy()
    active = margin > 0
    ambiguous = (margin == 0).any(axis=0)
    n = margin.shape[1]
    d_e, d_i = decays.dec_E, decays.dec_I
    coupling = np.full(n, np.nan)

    # Linearize within each sample's active set to obtain the local eigenvalues.
    if params.model_code == "rec":
        product = (params.w_IE.ravel()*params.w_EI.ravel()).detach().cpu().numpy().astype(float)
        coupling = (product[:, None]*active).sum(axis=0)
        discriminant = (d_e-d_i)**2-4*coupling
        root = np.sqrt(discriminant.astype(complex))
        first, second = (d_e+d_i+root)/2, (d_e+d_i-root)/2
        radius = np.maximum(np.abs(first), np.abs(second))
        if params.n_excitatory > 1:
            radius = np.maximum(radius, d_e)
        kind = np.where(
            discriminant < 0, "complex_pair",
            np.where(discriminant == 0, "repeated_root", "real_roots"),
        )
    else:
        first = np.full(n, d_e, dtype=complex)
        second = np.full(n, d_i if params.model_code == "ff" else np.nan, dtype=complex)
        radius = np.full(n, max(d_e, d_i) if params.model_code == "ff" else d_e)
        kind = np.full(n, "real_roots", dtype=object)

    reason = np.where(
        ambiguous, "relu_boundary",
        np.where(
            ~np.isfinite(radius), "nonfinite",
            np.where(
                radius >= 1, "noncontracting",
                np.where(kind == "repeated_root", "repeated_root", ""),
            ),
        ),
    )
    estimate = np.full(n, np.nan)
    valid = reason == ""
    estimate[valid] = np.log(tolerance)/np.log(radius[valid])
    return pd.DataFrame({"active_units": active.sum(axis=0), "relu_boundary": ambiguous,
        "coupling_k": coupling, "root_kind": kind, "lambda1_real": first.real,
        "lambda1_imag": first.imag, "lambda2_real": second.real, "lambda2_imag": second.imag,
        "dec_E": d_e, "dec_I": d_i if params.n_inhibitory else np.nan,
        "dec_E_multiplicity": params.n_excitatory-1 if params.model_code == "rec" else params.n_excitatory,
        "spectral_radius": radius, "mode_tolerance": tolerance,
        "mode_decay_steps": estimate, "undefined_reason": reason})


def fixed_point_detail(conditions, arrays, dataset):
    """Return final-model residual/mode details, a feature check and state arrays.

    Use the recorded analysis split; state arrays have shape (units, samples).
    Restores the recorded process-wide Torch numeric settings without resetting them.
    """
    config, device, train = _inputs(conditions, dataset)
    model = conditions["spec"]["model_code"]
    params = ModelParams.from_arrays(model, arrays["weights_trained"], device)
    split = conditions["analysis"]["split"]
    x = torch.as_tensor(dataset.arrays[f"x_{split}_small"].T, device=device)
    decays = Decays.from_taus(config.tau_e, config.tau_i)

    with torch.inference_mode():
        warm = warm_start_state(params, train, config)
        state = encode(params, x, config, warm)
        check = _probe_check(state.o_E, arrays["features"][f"h_{split}"].T, stage=f"final_{split}")

        # Compare the settled voltages with the fixed-point equations and one more step.
        drive = input_drive(params, x)
        gain_e, gain_i = 1/(1-decays.dec_E), 1/(1-decays.dec_I)
        if model == "rec":
            predicted_i = gain_i*(params.w_IE @ state.o_E)
        elif model == "ff":
            predicted_i = gain_i*(params.w_IIn @ x)
        else:
            predicted_i = None
        predicted_e = gain_e*(drive if predicted_i is None else drive-params.w_EI @ predicted_i)
        next_state = step(params, state, drive, decays, x)

    detail = local_modes(params, state, decays)
    detail["sample_index"] = np.arange(x.shape[1])
    detail["label"] = dataset.arrays[f"y_{split}"]
    detail["split"] = split
    detail["settle_steps"] = config.settle_steps
    detail["gain_E"] = gain_e
    detail["gain_I"] = gain_i if predicted_i is not None else np.nan
    detail["active_set_switch_next_step"] = (
        ((state.o_E > 0) != (next_state.o_E > 0)).any(dim=0).cpu().numpy()
    )

    for name, actual, predicted in (("E", state.v_E, predicted_e), ("I", state.v_I, predicted_i)):
        if predicted is None:
            detail[f"{name}_abs_error"] = np.nan
            detail[f"{name}_relative_denominator"] = np.nan
            detail[f"{name}_relative_error"] = np.nan
            detail[f"{name}_reason"] = "not_applicable"
            continue
        error = (predicted-actual).abs().amax(dim=0).cpu().numpy()
        denominator = actual.abs().amax(dim=0).cpu().numpy()
        detail[f"{name}_abs_error"] = error
        detail[f"{name}_relative_denominator"] = denominator
        detail[f"{name}_relative_error"] = np.divide(
            error, denominator, out=np.full_like(error, np.nan), where=denominator != 0,
        )
        detail[f"{name}_reason"] = np.where(
            denominator == 0, "zero_denominator",
            np.where(np.isfinite(error/np.maximum(denominator, 1e-30)), "", "nonfinite"),
        )

    states = {name: getattr(state, name).cpu().numpy() for name in ("v_E", "o_E", "v_I")}
    return detail, pd.DataFrame([check]), states
