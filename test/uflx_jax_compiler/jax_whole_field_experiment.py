"""Eager JAX experiment for whole-field ExternalOperator calls.

Run with pytest -n 0 test/uflx_jax_compiler/jax_whole_field_experiment.py.
The evaluator covers the real scalar operations used by this experiment.
"""

import operator

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.tree_util import Partial

from uflx.arithmetic import num
from uflx.calculus import coordinate, directional_derivative
from uflx.forms import apply, form, function
from uflx.nodes import Apply, DirectionalDerivative, Function, Integer, Product
from uflx_extensions.external_operators import (
    AbstractOperation,
    ExternalOperator,
    OutputType,
    external_operator,
)
from uflx_jax_compiler import trace

from ..uflx.spaces import space
from ..uflx_basix.integrals import D
from ..uflx_extensions.operations import Energy


def pointwise(fn, *operands):
    """Keep field parameters visible to JAX and delay sampling until a point is supplied."""
    if any(isinstance(arg, Partial) for arg in operands):

        def sample(values, point):
            return fn(*(v(point) if isinstance(v, Partial) else v for v in values))

        return Partial(sample, operands)
    return fn(*operands)


def sample_direction(field, changes, point):
    # A field can depend nonlinearly on its parameters. Differentiate its value at this point.
    return jax.jvp(lambda f: f(point), (field,), (changes,))[1]


def host_operand(value):
    if isinstance(value, Partial):
        return lambda point: np.asarray(value(np.asarray(point))).item()
    return np.asarray(value).item()


def external_call(operation, library):
    @jax.custom_jvp
    def call(*operands):
        return jnp.asarray(library[operation](*(host_operand(v) for v in operands)))

    @call.defjvp
    def jvp(primals, changes):
        value = call(*primals)
        # A tangent at a stationary point can still vary under an enclosing derivative.
        if all(not isinstance(dv, (Partial, jax.core.Tracer)) and dv == 0 for dv in changes):
            return value, jnp.zeros_like(value)
        directions = tuple(
            Partial(sample_direction, v, dv) if isinstance(v, Partial) else dv
            for v, dv in zip(primals, changes, strict=True)
        )
        action = operation.derivative_action()
        if action not in library:
            raise TypeError(f"missing evaluation for derivative action {action.identifier}")
        return value, external_call(action, library)(*primals, *directions)

    return call


def perturb(value, direction, epsilon):
    return value + epsilon * direction


def evaluate(node, env, library):
    def read(x):
        return evaluate(x, env, library)

    if isinstance(node, Function):
        return env[node]
    if isinstance(node, Integer):
        return float(node.value)
    if isinstance(node, Product):
        return pointwise(operator.mul, read(node.first), read(node.second))
    if isinstance(node, Apply):
        bindings = dict(zip(node.form.arguments, map(read, node.arguments), strict=True))
        return evaluate(node.form.expression, env | bindings, library)
    if isinstance(node, DirectionalDerivative):
        variables = node.bound()
        points = node.point if isinstance(node.point, tuple) else (node.point,)
        directions = node.direction if isinstance(node.direction, tuple) else (node.direction,)
        seeds = list(zip(variables, map(read, points), map(read, directions), strict=True))

        def perturbed(epsilon):
            bindings = {v: pointwise(perturb, x, h, epsilon) for v, x, h in seeds}
            return evaluate(node.expression, env | bindings, library)

        return jax.jvp(perturbed, (0.0,), (1.0,))[1]
    if isinstance(node, ExternalOperator):
        return external_call(node.operation, library)(*map(read, node.operands))
    raise TypeError(type(node).__name__)


class SampleProduct(AbstractOperation):
    identifier = "experiment.SampleProduct"
    order: int

    def output_type(self, operands):
        if len(operands) != 2 ** (self.order + 1):
            raise ValueError("expected a field and a scale for each primal and direction")
        return OutputType((), "real", None)

    def derivative_action(self):
        return SampleProduct(self.order + 1)


def library_bindings(saved=None):
    def remember(*fields):
        if saved is not None:
            saved.append(fields)

    def sample(field):
        return sum(field(np.array([x, 0.125])) for x in (0.125, 0.5))

    def value(field, a):
        remember(field)
        return a * sample(field)

    def first(field, a, dfield, da):
        remember(field, dfield)
        return a * sample(dfield) + da * sample(field)

    def second(field, a, dfield, da, hfield, ha, hdfield, hda):
        remember(field, dfield, hfield, hdfield)
        return a * sample(hdfield) + ha * sample(dfield) + da * sample(hfield) + hda * sample(field)

    return {SampleProduct(0): value, SampleProduct(1): first, SampleProduct(2): second}


def test_whole_field_against_oracle():
    V = space("cell field", domain=D, scalar_type="real")
    u, h, v = (function(V, name) for name in ("u", "h", "v"))
    a = function(space("scale", scalar_type="real"), "a")
    energy = external_operator(SampleProduct(0), (v * v, a))
    value = apply(form(energy, (v,)), (u,))
    first = directional_derivative(value, (u, a), (u, a), (h, num(3)))
    second = directional_derivative(first, (u, a), (u, a), (h, num(3)))

    def cell_field(offset):
        def field(p):
            assert isinstance(p, np.ndarray) and p.dtype != object
            return offset + p[0]

        return field

    cells = {0: cell_field(0), 1: cell_field(2)}
    direction = cell_field(1)
    # The graph oracle takes traceable fields. Only the host adapter under test
    # retains Python callbacks and checks that it supplies concrete NumPy points.
    x = coordinate(D)[0]
    library = library_bindings()
    retained = {}
    with jax.checking_leaks():
        for cell, field in cells.items():
            saved = []
            env = {u: Partial(field), h: Partial(direction), a: 2.0}
            for expr in (value, first, second):
                result = evaluate(expr, env, library_bindings(saved))
                assert float(result) == pytest.approx(
                    trace(expr, u, h, a, *library)(
                        num(2 * cell) + x, num(1) + x, 2.0, *library.values()
                    )
                )
            retained[cell] = saved

    # Both calls and all derivative traces have finished. Sample at fresh points, in reverse
    # cell order, including the mixed tangent field supplied to the second derivative action.
    for cell in (1, 0):
        assert {len(fields) for fields in retained[cell]} == {1, 2, 4}
        for x in (0.25, 0.375):
            point = np.array([x, 0.125])
            base, hval = cells[cell](point), direction(point)
            expected = (base**2, 2 * base * hval, 2 * base * hval, 2 * hval**2)
            for fields in retained[cell]:
                np.testing.assert_allclose([f(point) for f in fields], expected[: len(fields)])


def test_independent_operand_needs_no_action():
    operation = SampleProduct(0)
    library = {operation: lambda field, a: a * field(np.array([0.5]))}
    call = external_call(operation, library)
    fixed = Partial(lambda p: p[0])
    assert jax.jvp(lambda x: x + call(fixed, 2.0), (0.0,), (1.0,)) == (1.0, 1.0)


def test_stationary_scalar_operand_without_a_derivative_binding():
    a = function(space("scalar", scalar_type="real"), "a")
    expr = external_operator(Energy(), (a * a, num(2)))
    derivative = directional_derivative(expr, a, num(0), num(1))
    library = {Energy(): operator.mul}
    assert trace(derivative, *library)(*library.values()) == evaluate(derivative, {}, library) == 0
    active = directional_derivative(expr, a, num(1), num(1))
    with pytest.raises(TypeError, match="missing evaluation for derivative action"):
        evaluate(active, {}, library)


@pytest.mark.parametrize("transform", [jax.jit, jax.jacfwd])
def test_staged_and_batched_transforms_reach_the_python_boundary(transform):
    call = external_call(SampleProduct(0), library_bindings([]))
    field = Partial(lambda p: p[0])
    with pytest.raises(jax.errors.TracerArrayConversionError):
        transform(lambda a: call(field, a))(2.0)
