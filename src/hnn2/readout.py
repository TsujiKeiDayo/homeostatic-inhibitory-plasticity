"""Linear readout training: one ``nn.Linear`` classifier over frozen features (D10)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from .config import ExperimentConfig


@dataclass
class ReadoutHistory:
    """Per-optimizer-step training log; every field has one entry per step.

    ``batch_loss``/``batch_accuracy`` are measured before that step's update.
    ``l1_delta`` is how far the parameter vector moved that step. ``val_loss``/
    ``val_accuracy`` repeat the most recent scheduled validation result, and
    are NaN before the first one runs. ``val_evaluated`` marks fresh measurements.
    The unconditional final validation is stored separately in ReadoutResult,
    so the last history row may still contain an earlier measurement.
    """

    step: list[int] = field(default_factory=list)
    epoch: list[int] = field(default_factory=list)
    batch_loss: list[float] = field(default_factory=list)
    batch_accuracy: list[float] = field(default_factory=list)
    l1_delta: list[float] = field(default_factory=list)
    cumulative_l1: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_accuracy: list[float] = field(default_factory=list)
    val_evaluated: list[bool] = field(default_factory=list)


@dataclass
class ReadoutResult:
    """Outcome of one trained readout.

    ``best_val_accuracy`` is monitoring only — nothing is checkpointed, so the
    ``final_*`` scores use the last-epoch model, not the best model.
    ``final_test_accuracy_full`` is ``None`` unless full-test evaluation is
    enabled (D17).
    """

    history: ReadoutHistory
    best_val_accuracy: float
    final_val_accuracy: float
    final_test_accuracy: float
    final_test_accuracy_full: float | None
    n_features: int
    n_classes: int
    final_val_loss: float | None = None  # None only for legacy in-memory fixtures.
    final_step: int | None = None


def _evaluate(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    criterion: nn.Module,
    batch_size: int,
) -> tuple[float, float]:
    """Score ``x``/``y`` in mini-batches; return (mean loss, accuracy).

    Leaves the model in eval mode.
    """
    model.eval()
    total_loss, total_correct = 0.0, 0
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            xb, yb = x[start:start + batch_size], y[start:start + batch_size]
            logits = model(xb)
            # criterion averages over the batch, so re-weight by the batch size.
            total_loss += float(criterion(logits, yb)) * xb.shape[0]
            total_correct += int((logits.argmax(dim=1) == yb).sum())
    n = x.shape[0]
    return total_loss / n, total_correct / n


def _flat_params(model: nn.Module) -> torch.Tensor:
    """All parameters concatenated into one 1-D vector, detached from autograd."""
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def train_readout(
    features: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    config: ExperimentConfig,
    *,
    init_seed: int,
    shuffle_seed: int,
    device: torch.device,
) -> ReadoutResult:
    """Train one linear classifier on frozen features and report its accuracy.

    ``features`` contains ``h_train``, ``h_val`` and ``h_test`` arrays of shape
    (n_samples, n_features); ``labels`` contains row-aligned ``y_*`` arrays of
    nonnegative class indices. Optional ``h_test_full``/``y_test_full`` arrays
    enable full-test evaluation when configured.

    Only the classifier receives gradients. Standardisation uses train-split
    statistics, and final scores use the final model without early stopping.
    """
    def _x(key):
        arr = np.asarray(features[key], dtype=np.float32)
        # Shared train statistics prevent held-out information from leaking in.
        if config.readout_standardize:
            mu = features["h_train"].mean(axis=0, keepdims=True)
            sd = features["h_train"].std(axis=0, keepdims=True) + 1e-8
            arr = (arr - mu) / sd
        return torch.as_tensor(arr, dtype=torch.float32, device=device)

    def _y(key):
        return torch.as_tensor(np.asarray(labels[key], dtype=np.int64), device=device)

    # The bridge splits are small, so all three stay resident on the device and
    # mini-batching is plain indexing — no DataLoader.
    x_train, y_train = _x("h_train"), _y("y_train")
    x_val, y_val = _x("h_val"), _y("y_val")
    x_test, y_test = _x("h_test"), _y("y_test")

    n_features = int(x_train.shape[1])
    n_classes = int(max(int(y_train.max()), int(y_val.max()), int(y_test.max())) + 1)

    # Seeded CPU draw, so the starting weights never depend on unrelated earlier
    # torch randomness and are identical on CPU and GPU.
    torch_gen = torch.Generator(device="cpu").manual_seed(init_seed)
    model = nn.Linear(n_features, n_classes)
    with torch.no_grad():
        # uniform(-1/sqrt(fan_in), +1/sqrt(fan_in)) is PyTorch's own default range.
        bound = 1.0 / np.sqrt(n_features)
        model.weight.copy_(
            (torch.rand(model.weight.shape, generator=torch_gen) * 2 - 1) * bound
        )
        model.bias.copy_(
            (torch.rand(model.bias.shape, generator=torch_gen) * 2 - 1) * bound
        )
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.readout_lr)
    criterion = nn.CrossEntropyLoss()
    shuffle_rng = np.random.default_rng(shuffle_seed)

    history = ReadoutHistory()
    cumulative_l1 = 0.0
    # Baseline for the L1 path: step 1 is measured against the initial weights.
    previous_flat = _flat_params(model)
    best_val_accuracy = -np.inf
    val_loss = val_acc = float("nan")
    step = 0

    for epoch in range(config.readout_epochs):
        perm = shuffle_rng.permutation(x_train.shape[0])
        # Also restores train mode after a mid-epoch validation switched to eval.
        model.train()
        for start in range(0, len(perm), config.batch_size):
            idx = torch.as_tensor(perm[start:start + config.batch_size], device=device)
            xb, yb = x_train[idx], y_train[idx]

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            step += 1

            # Distance moved by this one update; the running sum is the CL5 path.
            current_flat = _flat_params(model)
            l1_delta = float((current_flat - previous_flat).abs().sum())
            previous_flat = current_flat
            cumulative_l1 += l1_delta

            # D11: validate every N steps (1 = the original per-batch behaviour).
            val_evaluated = step % max(int(config.readout_eval_every), 1) == 0
            if val_evaluated:
                val_loss, val_acc = _evaluate(
                    model, x_val, y_val, criterion, config.batch_size
                )
                best_val_accuracy = max(best_val_accuracy, val_acc)

            # Keep batch, parameter-path and validation histories step-aligned.
            history.step.append(step)
            history.epoch.append(epoch)
            history.batch_loss.append(float(loss))
            history.batch_accuracy.append(
                float((logits.argmax(dim=1) == yb).float().mean())
            )
            history.l1_delta.append(l1_delta)
            history.cumulative_l1.append(cumulative_l1)
            history.val_loss.append(float(val_loss))
            history.val_accuracy.append(float(val_acc))
            history.val_evaluated.append(val_evaluated)

    # Final scores use the last-epoch model — there is no best-checkpoint restore.
    final_val_loss, final_val_acc = _evaluate(
        model, x_val, y_val, criterion, config.batch_size
    )
    best_val_accuracy = max(best_val_accuracy, final_val_acc)
    # Test data is evaluated only after training has finished.
    _, final_test_acc = _evaluate(model, x_test, y_test, criterion, config.batch_size)

    # D17: evaluate the same final model on the optional full test split.
    final_test_acc_full = None
    if config.readout_final_eval_full_test and "h_test_full" in features:
        _, final_test_acc_full = _evaluate(
            model, _x("h_test_full"), _y("y_test_full"), criterion, config.batch_size
        )

    return ReadoutResult(
        history=history,
        best_val_accuracy=float(best_val_accuracy),
        final_val_accuracy=float(final_val_acc),
        final_test_accuracy=float(final_test_acc),
        final_test_accuracy_full=final_test_acc_full,
        n_features=n_features,
        n_classes=n_classes,
        final_val_loss=float(final_val_loss),
        final_step=step,
    )
