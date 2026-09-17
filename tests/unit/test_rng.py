import numpy as np
import pytest

from hnn2.rng import PRIMES, STREAMS, root_seed, stream, stream_torch_seed


def test_primes_match_original_study():
    """Original seed list: all primes in [10000, 20000]; index 0 -> 10007."""
    assert len(PRIMES) == 1033
    assert PRIMES[0] == 10007
    assert PRIMES[-1] == 19997
    assert root_seed(0) == 10007


def test_streams_are_deterministic():
    a = stream(0, "init/w_EIn").random(16)
    b = stream(0, "init/w_EIn").random(16)
    assert np.array_equal(a, b)


def test_streams_are_distinct_per_name_and_seed():
    base = stream(0, "init/w_EIn").random(16)
    assert not np.array_equal(base, stream(0, "shuffle").random(16))
    assert not np.array_equal(base, stream(1, "init/w_EIn").random(16))


def test_stream_independence():
    """Consuming one stream never shifts another (audit I-12 made impossible)."""
    shuffled_alone = stream(0, "shuffle").random(8)
    stream(0, "init/w_EIn").random(10_000)  # heavy use of a sibling stream
    assert np.array_equal(shuffled_alone, stream(0, "shuffle").random(8))


def test_unknown_stream_rejected():
    with pytest.raises(KeyError):
        stream(0, "init/w_EE")  # the old dead weight must not exist


def test_torch_seed_deterministic():
    assert stream_torch_seed(0, "readout/init") == stream_torch_seed(0, "readout/init")
    assert stream_torch_seed(0, "readout/init") != stream_torch_seed(1, "readout/init")


def test_stream_registry_is_append_only_contract():
    assert list(STREAMS.values()) == sorted(STREAMS.values())
    assert len(set(STREAMS.values())) == len(STREAMS)
