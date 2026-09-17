"""Seed tree (design D3, V2).

Every random draw descends from ``PRIMES[seed_index]`` through a *named* stream.
Streams are mutually independent: adding or removing a consumer of one never
shifts the draws of another.
"""

from __future__ import annotations

import numpy as np


def _sieve_primes(lo: int, hi: int) -> tuple[int, ...]:
    """The primes in ``[lo, hi]``, both ends inclusive."""
    flags = np.ones(hi + 1, dtype=bool)
    flags[:2] = False
    # Crossing out from i*i is safe: smaller multiples have a smaller prime factor.
    for i in range(2, int(hi ** 0.5) + 1):
        if flags[i]:
            flags[i * i :: i] = False
    return tuple(int(i) for i in np.nonzero(flags)[0] if i >= lo)


# Candidate root seeds, chosen by index so "seed 3" is a stable reference.
PRIMES: tuple[int, ...] = _sieve_primes(10000, 20000)

# Named streams. The integer is the SeedSequence spawn key — append only,
# never renumber (renumbering silently changes every derived draw).
STREAMS: dict[str, int] = {
    "init/w_EIn": 0,        # input -> E weights, Exp(1)/sqrt(n_in)
    "init/w_IIn": 1,        # input -> I weights; only `ff` needs them (V2)
    "shuffle": 2,           # batch order per homeostatic training epoch
    "readout/init": 3,      # linear readout weight init
    "readout/shuffle": 4,   # batch order while training the readout
    "mlp/init": 5,          # MLP baseline weight init (end-to-end reference)
    "mlp/shuffle": 6,       # batch order while training that MLP
    "mlp/readout_init": 7,  # readout placed on the MLP's frozen features
    "umap": 8,              # UMAP's stochastic embedding, qualitative figures
}


def root_seed(seed_index: int) -> int:
    """The root seed of one run. Raises ``ValueError`` if ``seed_index`` is out of range."""
    if not (0 <= seed_index < len(PRIMES)):
        raise ValueError(f"seed_index {seed_index} out of range [0, {len(PRIMES)})")
    return PRIMES[seed_index]


def stream(seed_index: int, name: str) -> np.random.Generator:
    """Create a fresh generator for one run's named stream.

    Repeating the same arguments restarts the sequence; retain the generator
    to consume successive draws. Different names use independent streams.
    Raises ``KeyError`` if ``name`` is not registered in ``STREAMS``.
    """
    if name not in STREAMS:
        raise KeyError(f"unknown stream {name!r}; registered: {sorted(STREAMS)}")
    seq = np.random.SeedSequence(entropy=root_seed(seed_index), spawn_key=(STREAMS[name],))
    return np.random.default_rng(seq)


def stream_torch_seed(seed_index: int, name: str) -> int:
    """Seed for torch consumers, drawn from the same named stream as ``stream``."""
    return int(stream(seed_index, name).integers(0, 2**63 - 1))
