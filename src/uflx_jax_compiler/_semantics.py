# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Numerical semantics of the expression nodes, evaluated with lexical bindings."""

from dataclasses import replace

import jax
import jax.numpy as jnp

from uflx.algorithms import operands as node_operands
from uflx.nodes import (
    Absolute,
    AbstractExpression,
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
from uflx_extensions.external_operators import ExternalOperator

from . import _finite_elements as elements
from . import _forms, _spatial, _tensors
from ._basis_tables import (
    keeps_quadrature_point,
    quadrature_tables,
    sampled_points,
    sampled_quadrature,
)
from ._context import BoundExpression, PointTabulation, lookup
from ._external import evaluate_external
from ._geometry import integrate, quadrature_sum


def needs_eager_samples(node, bindings):
    """A mapped predicate would trace both branches, including an unused binding."""
    pending = [node]
    seen = set()
    while pending:
        node = pending.pop()
        if node in seen:
            continue
        seen.add(node)
        binding = bindings.get(node)
        if isinstance(binding, BoundExpression):
            if binding.needs_eager_samples:
                return True
        elif isinstance(node, (Conditional, Form)):
            return True
        elif isinstance(node, Function):
            # A supplied expression or per-cell field may introduce its own conditional.
            field = bindings.get(node, bindings.get(f"coefficients:{node.label}"))
            if isinstance(field, (AbstractExpression, dict)):
                return True
        else:
            fields = [name for name in node.argument_names if name not in node.labels]
            pending.extend(reversed(node_operands(node, fields)))
    return False


def evaluate(node, bindings, site, arguments=()):
    """Evaluate a UFLx expression under lexical bindings at a local sample."""
    return _evaluate(node, bindings, site, arguments, {})


def _evaluate(node, bindings, site, arguments, memo):
    if site.tabulations and node.value_fields and not keeps_quadrature_point(node):
        # A whole-field operation can sample its operands anywhere, even if the
        # same field also has tabulated uses elsewhere in the integrand.
        site = replace(site, tabulations={})
        memo = {}

    def scoped_value(expr, slots=(), *, at=site, bind=None):
        # Callbacks can enter a new JAX trace even at the same sample point.
        local = bindings if bind is None else bindings | bind
        return evaluate(expr, local, at, slots)

    def value(expr, slots=()):
        # Form slots can contain arrays or callables. Reuse only the same tuple.
        if slots is not arguments:
            return scoped_value(expr, slots)
        if expr not in memo:
            memo[expr] = _evaluate(expr, bindings, site, arguments, memo)
        return memo[expr]

    def field(expr):
        return lambda sample: scoped_value(expr, arguments, at=sample)

    order = _tensors.axes(node, bindings)

    def operand(expr, slots=(), *, at=None, rank=len(node.value_shape)):
        data = value(expr, slots) if at is None else scoped_value(expr, slots, at=at)
        padding = rank - len(expr.value_shape)
        return _tensors.align(data, _tensors.axes(expr, bindings), order, padding)

    if order and 0 in _tensors.shape(node, bindings)[: len(order)]:
        return jnp.empty(_tensors.shape(node, bindings), dtype=_tensors.dtype(node))

    binding = bindings.get(node)
    if isinstance(binding, BoundExpression):
        return binding.read(site, arguments)

    def pointwise(*, batch=False):
        return _tensors.pointwise(
            node, bindings, lambda indices: scoped_value(node, arguments, bind=indices), batch=batch
        )

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
            at = site.at(points, tabulations=site.tabulations)

            def compare():
                with jax.ensure_compile_time_eval():
                    return jnp.real(operand(a, at=at)) < jnp.real(operand(b, at=at))

            # Make captured coefficient values explicit so outer AD stops at the predicate.
            predicate, inputs = jax.closure_convert(compare)
            return predicate(*map(jax.lax.stop_gradient, inputs))
        case Conditional(predicate, left, right):
            if _tensors.axes(predicate, bindings):
                return pointwise()
            condition = value(predicate)

            def branch(expr):
                result = jnp.asarray(operand(expr, arguments, at=site), dtype=_tensors.dtype(node))
                return jnp.broadcast_to(result, _tensors.shape(node, bindings))

            if not isinstance(condition, jax.core.Tracer):
                return branch(left if condition else right)
            return jax.lax.cond(condition, lambda: branch(left), lambda: branch(right))
        case Zero():
            return jnp.zeros(_tensors.shape(node, bindings), dtype=_tensors.dtype(node))
        case Sum(a, b):
            return operand(a, arguments) + operand(b, arguments)
        case Product(a, b):
            n = len(a.argument_spaces)
            return operand(a, arguments[:n]) * operand(b, arguments[n:])
        case Division(a, b):
            return jnp.divide(operand(a, arguments), operand(b))
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

            base = cast(operand(a, arguments))
            if isinstance(node, Power):
                if isinstance(node.exponent, Integer):
                    return base**node.exponent.value
                if isinstance(node.exponent, Float) and node.exponent.value == 0:
                    return jnp.ones_like(base)
            if node.scalar_type == "complex":
                # The principal argument of a negative real is +pi for either signed zero.
                # abs at zero preserves the imaginary tangent on the real axis.
                imag = base.imag
                base = jax.lax.complex(base.real, jnp.where(imag == 0, jnp.abs(imag), imag))
            if isinstance(node, Power):
                return jnp.power(base, cast(operand(node.exponent)))
            return jnp.sqrt(base) if isinstance(node, SquareRoot) else jnp.log(base)
        case Determinant(a) | Inverse(a):
            matrix_op = _tensors.determinant if isinstance(node, Determinant) else _tensors.inverse
            if a in site.shared_jacobians:
                matrix_op = (
                    _tensors.jacobian_determinant
                    if isinstance(node, Determinant)
                    else _tensors.jacobian_inverse
                )
            for _ in order:
                matrix_op = jax.vmap(matrix_op)
            return matrix_op(jnp.asarray(value(a)))
        case BroadcastTo(x, _):
            target = _tensors.shape(node, bindings)
            if 0 in target:
                return jnp.empty(target, dtype=_tensors.dtype(node))
            return jnp.broadcast_to(operand(x, arguments), target)
        case Tensor(entries):
            parts = [operand(x, arguments, rank=len(x.value_shape)) for x in entries]
            return jnp.stack(jnp.broadcast_arrays(*parts), axis=len(order))
        case Indexed(x, _):
            return _tensors.indexed(node, value(x, arguments), bindings)
        case ComponentTensor() | IndexSum():
            sampled = sampled_quadrature(node, bindings)
            if sampled is not None:
                expr, cell, points, weights = sampled
                rule = (value(points), value(weights))
                (q,) = node.indices

                def sample(at, row):
                    return scoped_value(expr, arguments, at=at, bind={q: row})

                return quadrature_sum(
                    node,
                    cell,
                    points,
                    rule,
                    sample,
                    expr,
                    site,
                    bindings,
                    scoped_value,
                    eager=q in dict(expr.free_indices) and needs_eager_samples(expr, bindings),
                )
            return _tensors.bind_indices(
                node, bindings, arguments, lambda x, local, slots: evaluate(x, local, site, slots)
            )
        case Function():
            return _forms.function_value(node, bindings, site, arguments, scoped_value)
        case Form(body, declarations):
            n = len(body.argument_spaces)
            return scoped_value(
                body, arguments[:n], bind=_forms.form_bindings(declarations, arguments[n:])
            )
        case Apply(operator, operands):
            if order:
                return pointwise()
            slots = tuple(
                _forms.supplied(x, s, site, scoped_value)
                for x, s in zip(operands, operator.argument_spaces, strict=True)
            )
            return value(operator, slots)
        case ExternalOperator():
            if order:
                return pointwise()
            return evaluate_external(node, bindings, site, scoped_value, memo)
        case DirectionalDerivative():
            reference = site.require_reference("DirectionalDerivative")
            if order:
                return pointwise()
            return reference.directional(node, arguments, scoped_value)
        case Coordinate(domain):
            return jnp.asarray(site.points[domain], dtype=float)
        case Gradient(x, domain):
            point = jnp.asarray(site.points[domain])
            if not point.size:
                return jnp.empty(_tensors.shape(node, bindings), dtype=_tensors.dtype(node))
            if site.reference is None:
                return elements.tabulated_gradient(node, bindings, site, scoped_value)
            return site.reference.gradient(x, domain, arguments, site, scoped_value, point)
        case Divergence(x, domain):
            return jnp.trace(value(Gradient(x, domain), arguments), axis1=-2, axis2=-1)
        case Curl(x, domain):
            g = value(Gradient(x, domain), arguments)
            if g.shape[-1] == 2:
                return g[..., 1, 0] - g[..., 0, 1]
            return jnp.stack(
                (
                    g[..., 2, 1] - g[..., 1, 2],
                    g[..., 0, 2] - g[..., 2, 0],
                    g[..., 1, 0] - g[..., 0, 1],
                ),
                axis=-1,
            )
        case Integral(x, _):
            return integrate(node, bindings, field(x), site, scoped_value)
        case PullBack(x, mapping):
            if _tensors.axes(mapping, bindings):
                return pointwise(batch=not needs_eager_samples(node, bindings))
            at = _spatial.pullback_site(mapping, site, scoped_value)
            if (source := sampled_points(mapping)) is not None:
                points, q, cell = source
                tables = quadrature_tables(
                    points, cell, bindings, _tensors.size(points.value_shape[0], bindings)
                )
                if tables:
                    point = at.points[cell]
                    rows = {key: data[bindings[q]] for key, data in tables.items()}
                    at = at.at({cell: point}, tabulations={cell: PointTabulation(point, rows)})
            return scoped_value(x, arguments, at=at)
        case PushForward(x, mesh, law):
            reference = site.require_reference("PushForward")
            return reference.push_forward(field(x), mesh, law, site, scoped_value)
        case Conormal(side, conormals):
            reference = site.require_reference("Conormal")
            return reference.conormal(side, conormals, site, scoped_value)
        case _ if elements.is_basix_basis(node):
            return elements.basis_value(node, site)
        case AbstractExpression(is_terminal=True, domain=None, argument_spaces=(), free_indices=()):
            return jnp.asarray(lookup(bindings, node))
    raise TypeError(f"the reference interpreter has no evaluation rule for {type(node).__name__}")
