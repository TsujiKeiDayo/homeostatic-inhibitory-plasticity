"""Metrics describing how the network represents its inputs, not how well.

The input is usually ``o_E``, shape ``(n_units, n_samples)``, read along either
of two axes: *lifetime* is one unit across stimuli (``axis=1``), *population*
is one stimulus across units (``axis=0``).
"""
