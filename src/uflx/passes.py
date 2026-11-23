"""Evaluation passes and the standard lowering recipe."""

from ._src.passes.applications import expand_applications
from ._src.passes.arithmetic import expand_arithmetic
from ._src.passes.derivatives import apply_derivatives, expand_derivatives_with
from ._src.passes.gradients import apply_gradients
from ._src.passes.integrals import expand_integrals
from ._src.passes.lowering import lower
from ._src.passes.pullbacks import expand_pull_backs

__all__ = [
    "apply_derivatives",
    "apply_gradients",
    "expand_applications",
    "expand_arithmetic",
    "expand_derivatives_with",
    "expand_integrals",
    "expand_pull_backs",
    "lower",
]
