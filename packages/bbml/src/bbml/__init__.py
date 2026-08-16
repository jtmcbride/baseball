"""Feature builders, datasets, training, and serving.

Populated in M2. The load-bearing constraint recorded here so it is not lost:
the feature builder is ONE function with TWO callers (batch over Parquet, live
over an MLB game-feed state object), enforced by a golden parity test. Divergence
between the two is the failure mode that makes a live model degrade in ways that
are nearly impossible to diagnose after the fact.
"""

__version__ = "0.1.0"
