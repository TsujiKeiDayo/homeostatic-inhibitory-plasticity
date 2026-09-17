"""Rate-based network (continuous firing rates, not spikes) obeying Dale's law.

Every neuron is excitatory (E) or inhibitory (I), so weights are stored
non-negative and inhibition is subtracted in ``dynamics`` rather than carried
as a negative value.
"""
