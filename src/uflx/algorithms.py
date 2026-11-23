"""Traverse, compare and rewrite UFLx expression graphs."""

from ._src.algorithms.binding import alpha_equivalent
from ._src.algorithms.names import derived_index, derived_indices
from ._src.algorithms.rewriting import complete_indices
from ._src.algorithms.substitution import replace, substitute_values
from ._src.algorithms.traversal import bottom_up, map_fields, map_values, memoized
from ._src.nodes.static import free_indices
from ._src.nodes.structure import expression_operands, operands

__all__ = [
    "alpha_equivalent",
    "bottom_up",
    "complete_indices",
    "derived_index",
    "derived_indices",
    "expression_operands",
    "free_indices",
    "map_fields",
    "map_values",
    "memoized",
    "operands",
    "replace",
    "substitute_values",
]
