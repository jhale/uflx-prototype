from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from ..substitution import _freshen, _map_values

if TYPE_CHECKING:
    from ..nodes.base import AbstractExpression
    from ..types import _Construction


# Shared implementation helpers; the package public API is declared in __init__.py.
__all__ = ["_commute"]


def _commute(
    f: AbstractExpression,
    evaluate: Callable[[AbstractExpression], AbstractExpression],
    avoid: Sequence[_Construction],
) -> AbstractExpression:
    # Move an evaluated node into its commuting operands, after making a binding node fresh.
    fresh = _freshen(f, avoid)
    return _map_values(fresh, evaluate, fresh.commuting)
