"""Pointwise library calls and whole-field external operators."""

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import Partial

from uflx.calculus import coordinate

from ._context import bound
from ._forms import supplied
from ._tensors import size


@dataclass(frozen=True)
class Pointwise:
    """Apply a library function to operand values. Set traceable for a pure JAX function."""

    function: Callable
    traceable: bool = False


def _converted_field(fn, point):
    # Make the coefficients of the field explicit across the custom derivative boundary.
    converted, coefficients = jax.closure_convert(fn, point)
    return Partial(lambda values, p: converted(p, *values), tuple(coefficients))


def _sample_direction(field, changes, point):
    return jax.jvp(lambda f: f(point), (field,), (changes,))[1]


def _zero_direction(changes):
    return all(
        not isinstance(v, jax.core.Tracer)
        and (jnp.asarray(v).dtype == jax.dtypes.float0 or np.all(np.asarray(v) == 0))
        for v in jax.tree.leaves(changes)
    )


def _pointwise_action(action, primals, changes, value):
    # A host callback has no transpose. Evaluate its linear action on basis directions
    # so JAX can transpose the contraction, including for real-linear complex operations.
    zero = tuple(jnp.zeros_like(v) for v in primals)
    tangent = jnp.zeros_like(value)
    for i, (p, dp) in enumerate(zip(primals, changes, strict=True)):
        if jnp.asarray(dp).dtype == jax.dtypes.float0:
            continue
        for index in np.ndindex(jnp.shape(p)):
            unit = zero[i].at[index].set(1)
            component = jnp.asarray(dp)[index]
            for scale in (1, 1j) if jnp.iscomplexobj(p) else (1,):
                basis = (*zero[:i], scale * unit, *zero[i + 1 :])
                column = action(*primals, *basis)
                weight = jnp.real(component) if scale == 1 else jnp.imag(component)
                tangent = tangent + column * weight
    return tangent


def evaluate_external(node, frame):
    operation = node.operation
    binding = frame.environment[operation]
    pointwise = isinstance(binding, Pointwise)
    if pointwise:
        inputs = [frame.value(x) for x in node.operands]
    else:
        inputs = [bound(supplied(x, x, frame), frame) for x in node.operands]

    sample_point = frame.points.get(node.domain) if not pointwise else None

    def result(fn, args, point):
        value = fn(*args)
        if callable(value) and node.domain is not None:
            value = value(point)
        return value

    def has_action(op):
        try:
            return op.derivative_action() in frame.environment
        except TypeError:
            return False

    if (not pointwise or binding.traceable) and not has_action(operation):
        return jnp.asarray(result(binding.function if pointwise else binding, inputs, sample_point))

    if not pointwise:
        for i, (value, operand) in enumerate(zip(inputs, node.operands, strict=True)):
            if callable(value):
                point = jnp.zeros(coordinate(operand.domain).value_shape)
                inputs[i] = _converted_field(value, point)

    dtype = {"real": float, "complex": complex, "boolean": bool}[node.scalar_type]
    shape = tuple(size(n, frame) for n in node.value_shape)
    signature = jax.ShapeDtypeStruct(shape, jnp.result_type(dtype))

    def evaluate(op, sample_point, *operands):
        impl = frame.environment[op]

        @jax.custom_jvp
        def call(point, *args):
            if isinstance(impl, Pointwise) and impl.traceable:
                return jnp.asarray(impl.function(*args), dtype=dtype)
            if isinstance(impl, Pointwise):

                def host(*values):
                    return np.asarray(impl.function(*values), dtype=signature.dtype)

                if any(isinstance(v, jax.core.Tracer) for v in args):
                    return jax.pure_callback(host, signature, *args, vmap_method="sequential")
                return jnp.asarray(host(*args))
            return jnp.asarray(result(impl, args, point), dtype=dtype)

        @call.defjvp
        def derivative(primals, changes):
            point, *primals = primals
            dpoint, *changes = changes
            value = call(point, *primals)
            spatial = jnp.zeros_like(value)
            if point is not None and not _zero_direction(dpoint):
                spatial = jax.jvp(lambda p: result(impl, primals, p), (point,), (dpoint,))[1]
            if _zero_direction(changes):
                return value, spatial

            action = op.derivative_action()
            if action not in frame.environment:
                raise TypeError(f"missing evaluation for derivative action {action.identifier}")
            action_binding = frame.environment[action]
            if pointwise and isinstance(action_binding, Pointwise) and not action_binding.traceable:
                tangent = _pointwise_action(
                    lambda *args: evaluate(action, point, *args), primals, changes, value
                )
                return value, tangent
            directions = tuple(
                Partial(_sample_direction, p, dp) if isinstance(p, Partial) else dp
                for p, dp in zip(primals, changes, strict=True)
            )
            return value, spatial + evaluate(action, point, *primals, *directions)

        return call(sample_point, *operands)

    return evaluate(operation, sample_point, *inputs)
