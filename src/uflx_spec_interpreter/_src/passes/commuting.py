from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from ..substitution import freshen, map_values

if TYPE_CHECKING:
    from ..nodes.base import AbstractExpression
    from ..types import Construction


def commute(
    f: AbstractExpression,
    evaluate: Callable[[AbstractExpression], AbstractExpression],
    avoid: Sequence[Construction],
) -> AbstractExpression:
    # Move an evaluated node into its commuting operands, after making a binding node fresh.
    fresh = freshen(f, avoid)
    return map_values(fresh, evaluate, fresh.commuting)
