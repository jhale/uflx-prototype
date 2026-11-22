"""Numerical semantics of the expression nodes, evaluated with lexical bindings."""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from uflx._src.extensions.external_operators import ExternalOperator
from uflx.nodes import (
    Absolute,
    Apply,
    Boolean,
    BroadcastTo,
    Complex,
    ComponentTensor,
    Conditional,
    Conjugate,
    Conormal,
    Coordinate,
    Curl,
    Determinant,
    DirectionalDerivative,
    Divergence,
    Division,
    Float,
    Form,
    Function,
    Gradient,
    ImaginaryPart,
    Indexed,
    IndexSum,
    Integer,
    Integral,
    Inverse,
    LessThan,
    Logarithm,
    Power,
    Product,
    PullBack,
    PushForward,
    RealPart,
    SquareRoot,
    Sum,
    Tensor,
    Zero,
)
from uflx_basix.nodes import RuntimeArray, VertexCoordinates

from . import _finite_elements as elements
from . import _forms, _spatial, _tensors
from ._external import evaluate_external
from ._geometry import integrate


@dataclass(frozen=True, eq=False)
class Frame:
    """Evaluate with fixed function bindings, index assignments and spatial points."""

    environment: dict
    indices: dict
    points: dict
    cells: dict
    comparison: bool = False
    _memo: dict = field(default_factory=dict, repr=False)

    def value(self, node, arguments=()):
        if arguments:
            return interpret(node, self, arguments)
        # Only the node's free indices can affect its value at this point.
        key = (node, tuple(self.indices.get(i) for i, _ in node.free_indices))
        if any(hasattr(i, "shape") for i in jax.tree.leaves(key[1])):
            return interpret(node, self, arguments)
        if key not in self._memo:
            result = interpret(node, self, ())
            self._memo[key] = jax.lax.stop_gradient(result) if self.comparison else result
        return self._memo[key]

    def at(self, points, cells=None):
        return Frame(
            self.environment,
            self.indices,
            self.points | points,
            self.cells if cells is None else self.cells | cells,
            self.comparison,
        )

    def sampled_in(self, caller):
        """Keep lexical bindings and use the points and cells at the sampling site."""
        if self.points is caller.points and self.cells is caller.cells:
            return self
        return Frame(self.environment, self.indices, caller.points, caller.cells, caller.comparison)

    def bind_arguments(self, bindings):
        return Frame(
            self.environment | bindings, self.indices, self.points, self.cells, self.comparison
        )

    def bind_indices(self, indices):
        return Frame(
            self.environment,
            self.indices | indices,
            self.points,
            self.cells,
            self.comparison,
            self._memo,
        )

    def input(self, node, label):
        """Library data may be supplied by node or by its runtime label."""
        return self.environment[node] if node in self.environment else self.environment[label]


def interpret(node, frame, arguments):
    binding = frame.environment.get(node)
    if isinstance(binding, _forms.ExpressionBinding):
        return binding.read(frame, arguments)

    value = frame.value
    match node:
        case Boolean(v) | Float(v):
            return v
        case Integer(v):
            # UFLx integer literals are real scalars, including in coefficient fractions.
            return float(v)
        case Complex(a, b):
            return complex(a.value, b.value)
        case LessThan(a, b):
            points = {k: jax.lax.stop_gradient(p) for k, p in frame.points.items()}
            local = Frame(frame.environment, frame.indices, points, frame.cells, comparison=True)
            return jnp.real(local.value(a)) < jnp.real(local.value(b))
        case Conditional(predicate, left, right):
            condition = value(predicate)
            dtype = {"boolean": bool, "real": float, "complex": complex}[node.scalar_type]

            def branch(expr):
                local = frame.bind_arguments({})
                return jnp.asarray(local.value(expr, arguments), dtype=dtype)

            if not isinstance(condition, jax.core.Tracer):
                return jnp.asarray(value(left if condition else right, arguments), dtype=dtype)
            # Branch-local frames keep intermediate tracers inside their own branch.
            return jax.lax.cond(condition, lambda: branch(left), lambda: branch(right))
        case Zero(shape, _, _, scalar_type):
            dtype = complex if scalar_type == "complex" else float
            return (
                jnp.zeros(tuple(_tensors.size(n, frame) for n in shape), dtype=dtype)
                if shape
                else dtype(0)
            )
        case Sum(a, b):
            return value(a, arguments) + value(b, arguments)
        case Product(a, b):
            n = len(a.argument_spaces)
            return value(a, arguments[:n]) * value(b, arguments[n:])
        case Division(a, b):
            return jnp.divide(value(a, arguments), value(b))
        case Conjugate(a):
            return jnp.conj(value(a, tuple(_forms.conjugated(v) for v in arguments)))
        case RealPart(a):
            return jnp.real(value(a, arguments))
        case ImaginaryPart(a):
            return jnp.imag(value(a, arguments))
        case Absolute(a):
            return jnp.abs(value(a, arguments))
        case SquareRoot(a) | Logarithm(a) | Power(a, _):

            def cast(v):
                if node.scalar_type == "complex":
                    return jnp.asarray(v, dtype=complex)
                return jnp.asarray(jnp.real(v), dtype=float)

            base = cast(value(a, arguments))
            if isinstance(node, Power):
                if isinstance(node.exponent, Integer):
                    return base**node.exponent.value
                if isinstance(node.exponent, Float) and node.exponent.value == 0:
                    return jnp.ones_like(base)
                return jnp.power(base, cast(value(node.exponent)))
            return jnp.sqrt(base) if isinstance(node, SquareRoot) else jnp.log(base)
        case Determinant(a):
            return _tensors.determinant(value(a))
        case Inverse(a):
            return _tensors.inverse(jnp.asarray(value(a)))
        case BroadcastTo(x, shape):
            target = tuple(_tensors.size(n, frame) for n in shape)
            if 0 in target:
                dtype = {"boolean": bool, "complex": complex, "real": float}[node.scalar_type]
                return jnp.empty(target, dtype=dtype)
            return jnp.broadcast_to(value(x, arguments), target)
        case Tensor(entries):
            return jnp.stack([value(entry, arguments) for entry in entries])
        case Indexed(x, indices):
            key = _tensors.flat(_tensors.index_value(i, frame.indices) for i in indices)
            return jnp.asarray(value(x, arguments))[key]
        case ComponentTensor() | IndexSum():
            return _tensors.bound_tensor(node, frame, arguments)
        case Function():
            return _forms.function_value(node, frame, arguments)
        case Form(body, declarations):
            return _forms.form_value(body, declarations, frame, arguments)
        case Apply(operator, operands):
            slots = tuple(
                _forms.supplied(x, s, frame)
                for x, s in zip(operands, operator.argument_spaces, strict=True)
            )
            return value(operator, slots)
        case ExternalOperator():
            return evaluate_external(node, frame)
        case DirectionalDerivative(body, variables, point, direction):
            return _forms.directional(body, variables, point, direction, frame, arguments)
        case Coordinate(domain):
            return jnp.asarray(frame.points[domain], dtype=float)
        case Gradient(x, domain):
            return _spatial.gradient(x, domain, frame, arguments)
        case Divergence(x, domain):
            return jnp.trace(_spatial.gradient(x, domain, frame, arguments), axis1=-2, axis2=-1)
        case Curl(x, domain):
            return _spatial.curl(x, domain, frame, arguments)
        case Integral(x, domain):
            return integrate(domain, lambda local: local.value(x, arguments), frame)
        case PullBack(x, mapping):
            return _spatial.pull_back(x, mapping, frame, arguments)
        case PushForward(x, mesh, law):
            return _spatial.push_forward(x, mesh, law, frame, arguments)
        case Conormal(side, conormals):
            return _spatial.conormal(side, conormals, frame)
        case RuntimeArray():
            return jnp.asarray(frame.input(node, node.label))
        case VertexCoordinates(topology, vertices):
            return elements.vertex_value(topology, vertices, frame)
        case _ if elements.is_basix_basis(node):
            return elements.basis_value(node, frame)
    raise TypeError(f"the reference interpreter has no evaluation rule for {type(node).__name__}")
