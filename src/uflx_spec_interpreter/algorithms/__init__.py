# SPDX-License-Identifier: MIT
"""Explicit structural algorithms for consumers of UFLx expressions.

Alpha fingerprints find candidate matches; use alpha_equivalent to confirm them.
These utilities do not change ordinary expression equality, hashing or interning.
"""

from .._src.algorithms.alpha import AlphaHasher, alpha_hash
from .._src.algorithms.binding import alpha_equivalent

__all__ = ["AlphaHasher", "alpha_equivalent", "alpha_hash"]
