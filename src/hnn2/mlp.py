"""One-hidden-layer MLP baseline trained end-to-end with backprop (C8)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .config import ExperimentConfig


class OneHiddenMLP(nn.Module):
    """Input -> ReLU hidden layer -> linear output."""

    def __init__(self, n_in: int, hidden_dim: int, n_out: int):
        super().__init__()
        self.hidden = nn.Linear(n_in, hidden_dim)
        self.readout = nn.Linear(hidden_dim, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Class logits; :meth:`encode` gives the representation instead."""
        return self.readout(torch.relu(self.hidden(x)))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """The representation: post-ReLU hidden activations, (n_samples, hidden_dim)."""
        return torch.relu(self.hidden(x))


def _seeded_init(model: OneHiddenMLP, seed: int) -> None:
    """Re-initialise both layers in place from an explicitly seeded CPU generator.

    Weights are reproducible across devices and independent of prior global RNG
    use; the sampling range still matches PyTorch's default for ``nn.Linear``.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for layer in (model.hidden, model.readout):
            # weight is (fan_out, fan_in), so shape[1] is the input width.
            bound = 1.0 / np.sqrt(layer.weight.shape[1])
            layer.weight.copy_(
                (torch.rand(layer.weight.shape, generator=gen) * 2 - 1) * bound
            )
            layer.bias.copy_(
                (torch.rand(layer.bias.shape, generator=gen) * 2 - 1) * bound
            )


def train_mlp_e2e(
    x_train: np.ndarray,
    y_train: np.ndarray,
    config: ExperimentConfig,
    *,
    init_seed: int,
    shuffle_seed: int,
    device: torch.device,
) -> OneHiddenMLP:
    """Train a whole network end-to-end from raw input; return it in eval mode.

    ``x_train`` is (n_samples, n_features); ``y_train`` is a row-aligned vector
    of nonnegative class indices.

    Every layer receives gradients, unlike ``readout.train_readout``, which
    fits only a linear classifier over frozen features. Train split only: no
    validation and no early stopping, just a fixed epoch budget from ``config``.
    """
    n_classes = int(y_train.max()) + 1
    # Hidden width == excitatory population size, matching the network arms' o_E.
    model = OneHiddenMLP(x_train.shape[1], config.n_excitatory, n_classes)
    _seeded_init(model, init_seed)
    model = model.to(device)

    # The whole training split fits on the device, so mini-batching is indexing.
    x = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    y = torch.as_tensor(y_train, dtype=torch.int64, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.readout_lr)
    criterion = nn.CrossEntropyLoss()
    rng = np.random.default_rng(shuffle_seed)

    model.train()
    for _ in range(config.readout_epochs):
        perm = rng.permutation(x.shape[0])
        for start in range(0, len(perm), config.batch_size):
            idx = torch.as_tensor(perm[start:start + config.batch_size], device=device)
            optimizer.zero_grad()
            loss = criterion(model(x[idx]), y[idx])
            loss.backward()
            optimizer.step()

    model.eval()
    return model
