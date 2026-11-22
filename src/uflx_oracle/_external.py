"""Evaluation and declared derivative actions for library-owned operators."""

from collections.abc import Callable
from dataclasses import dataclass

from . import _dual as ad
from ._context import call_at
from ._forms import supplied


@dataclass(frozen=True)
class Pointwise:
    """Bind an ExternalOperator to a callable of operand values at the current point."""

    function: Callable


def parts(operand, tag):
    if callable(operand):

        def component(index):
            def read(*args):
                return parts(operand(*args), tag)[index]

            read._derivative_tags = lambda: ad.tags(operand) - {tag}
            return read

        return component(0), component(1)
    return ad.split(operand, tag)


def evaluate_external(node, frame):
    operation = node.operation
    if isinstance(frame.environment[operation], Pointwise):
        inputs = tuple(frame.value(operand) for operand in node.operands)
    else:
        inputs = tuple(supplied(operand, operand, frame) for operand in node.operands)

    def evaluate(operation, inputs):
        active = set().union(*(ad.tags(operand) for operand in inputs))
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
        result = call_at(frame.points, fn, *inputs)
        if node.domain is not None and callable(result):
            return call_at(frame.points, ad.call_field, result, frame.points[node.domain])
        return ad.normalize(result)

    return evaluate(operation, inputs)
