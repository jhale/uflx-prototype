"""Numerical semantics of the expression nodes, evaluated with lexical bindings."""

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
    Mesh,
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
from ._context import BoundExpression, lookup
from ._external import evaluate_external
from ._geometry import integrate, tangent_projector


def evaluate(node, bindings, site, arguments=()):
    """Evaluate a UFLx expression under lexical bindings at a local sample."""

    def value(expr, slots=(), *, at=site, bind=None):
        local = bindings if bind is None else bindings | bind
        return evaluate(expr, local, at, slots)

    def field(expr):
        return lambda sample: value(expr, arguments, at=sample)

    binding = bindings.get(node)
    if isinstance(binding, BoundExpression):
        return binding.read(site, arguments)

    match node:
        case Boolean(v) | Float(v):
            return v
        case Integer(v):
            # UFLx integer literals are real scalars, including in coefficient fractions.
            return float(v)
        case Complex(a, b):
            return complex(a.value, b.value)
        case LessThan(a, b):
            # Use the primal inputs for the comparison. A derivative written
            # inside an operand still has its ordinary mathematical meaning.
            points = {d: jax.lax.stop_gradient(p) for d, p in site.points.items()}
            at = site.at(points)

            def compare():
                with jax.ensure_compile_time_eval():
                    return jnp.real(value(a, at=at)) < jnp.real(value(b, at=at))

            # Make captured coefficient values explicit so outer AD stops at the predicate.
            predicate, inputs = jax.closure_convert(compare)
            return predicate(*map(jax.lax.stop_gradient, inputs))
        case Conditional(predicate, left, right):
            condition = value(predicate)
            dtype = {"boolean": bool, "real": float, "complex": complex}[node.scalar_type]

            def branch(expr):
                return jnp.asarray(value(expr, arguments), dtype=dtype)

            if not isinstance(condition, jax.core.Tracer):
                return branch(left if condition else right)
            return jax.lax.cond(condition, lambda: branch(left), lambda: branch(right))
        case Zero(shape, _, _, scalar_type):
            dtype = complex if scalar_type == "complex" else float
            return (
                jnp.zeros(tuple(_tensors.size(n, bindings) for n in shape), dtype=dtype)
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
            target = tuple(_tensors.size(n, bindings) for n in shape)
            if 0 in target:
                dtype = {"boolean": bool, "complex": complex, "real": float}[node.scalar_type]
                return jnp.empty(target, dtype=dtype)
            return jnp.broadcast_to(value(x, arguments), target)
        case Tensor(entries):
            return jnp.stack([value(entry, arguments) for entry in entries])
        case Indexed(x, indices):
            key = _tensors.flat(_tensors.index_value(i, bindings) for i in indices)
            return jnp.asarray(value(x, arguments))[key]
        case ComponentTensor() | IndexSum():
            return _tensors.bound_tensor(
                node, bindings, lambda indices: value(node.expression, arguments, bind=indices)
            )
        case Function():
            return _forms.function_value(node, bindings, site, arguments, value)
        case Form(body, declarations):
            n = len(body.argument_spaces)
            return value(
                body, arguments[:n], bind=_forms.form_bindings(declarations, arguments[n:])
            )
        case Apply(operator, operands):
            slots = tuple(
                _forms.supplied(x, s, site, value)
                for x, s in zip(operands, operator.argument_spaces, strict=True)
            )
            return value(operator, slots)
        case ExternalOperator():
            return evaluate_external(node, bindings[node.operation], site, value)
        case DirectionalDerivative():
            return _forms.directional(node, arguments, value)
        case Coordinate(domain):
            return jnp.asarray(site.points[domain], dtype=float)
        case Gradient(x, domain):
            point = jnp.asarray(site.points[domain])
            if not point.size:
                shape = tuple(_tensors.size(n, bindings) for n in x.value_shape)
                return jnp.empty(
                    (*shape, 0), dtype=complex if x.scalar_type == "complex" else float
                )

            def at_point(p):
                result = value(x, arguments, at=site.at({domain: p}))
                return jnp.asarray(result, dtype=jnp.result_type(result, 0.0))

            g = jax.jacfwd(at_point)(point)
            if isinstance(domain, Mesh) and domain.dimension < point.size:
                g = g @ tangent_projector(domain, site, value)
            return g
        case Divergence(x, domain):
            return jnp.trace(value(Gradient(x, domain), arguments), axis1=-2, axis2=-1)
        case Curl(x, domain):
            return _spatial.curl(value(Gradient(x, domain), arguments))
        case Integral(x, domain):
            return integrate(domain, field(x), site, value)
        case PullBack(x, mapping):
            return value(x, arguments, at=_spatial.pullback_site(mapping, site, value))
        case PushForward(x, mesh, law):
            return _spatial.push_forward(field(x), mesh, law, site, value)
        case Conormal(side, conormals):
            return _spatial.conormal(side, conormals, site, value)
        case RuntimeArray():
            return jnp.asarray(lookup(bindings, node, node.label))
        case VertexCoordinates(topology, vertices):
            return elements.vertex_value(topology, vertices, site)
        case _ if elements.is_basix_basis(node):
            return elements.basis_value(node, site)
    raise TypeError(f"the reference interpreter has no evaluation rule for {type(node).__name__}")
