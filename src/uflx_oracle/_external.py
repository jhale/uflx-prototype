"""Evaluation and declared derivative actions for library-owned operators."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from . import _dual as ad
from ._context import call, parts, sample, tags
from ._forms import supplied


@dataclass(frozen=True)
class Pointwise:
    """Bind an ExternalOperator to a callable of operand values at the current point."""

    function: Callable


def evaluate_external(node, frame):
    operation = node.operation
    if isinstance(frame.environment[operation], Pointwise):
        inputs = tuple(frame.value(operand) for operand in node.operands)
    else:
        inputs = tuple(supplied(operand, operand, frame) for operand in node.operands)

    def evaluate(operation, inputs):
        active = tags(inputs)
        if active:
            tag = max(active)
            values, directions = zip(*(parts(operand, tag) for operand in inputs), strict=True)
            primal = evaluate(operation, values)
            if all(ad.is_zero(direction) for direction in directions):
                return primal
            action = operation.derivative_action()
            if action not in frame.environment:
                raise TypeError(f"missing evaluation for derivative action {action.identifier}")
            tangent = evaluate(action, values + directions)
            return ad.pair(tag, primal, tangent)

        binding = frame.environment[operation]
        fn = binding.function if isinstance(binding, Pointwise) else binding
        result = call(frame, fn, *inputs)
        if node.domain is not None and callable(result):
            return sample(result, frame, node.domain)
        return np.asarray(result) if isinstance(result, (list, tuple)) else result

    return evaluate(operation, inputs)
