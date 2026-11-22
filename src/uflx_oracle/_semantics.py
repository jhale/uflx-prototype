"""Numerical semantics of the expression nodes, evaluated with lexical bindings."""

import numpy as np

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
from ._geometry import integrate


def interpret(node, frame, arguments):
    value = frame.value
    match node:
        case Boolean(v) | Integer(v) | Float(v):
            return v
        case Complex(a, b):
            return complex(a.value, b.value)
        case LessThan(a, b):
            return np.real(value(a)) < np.real(value(b))
        case Conditional(predicate, left, right):
            result = value(left if value(predicate) else right, arguments)
            return np.asarray(result, dtype=complex) if node.scalar_type == "complex" else result
        case Zero(shape, _, _, scalar_type):
            dtype = complex if scalar_type == "complex" else float
            return (
                np.zeros(tuple(_tensors.size(n, frame) for n in shape), dtype=dtype)
                if shape
                else dtype(0)
            )
        case Sum(a, b):
            return value(a, arguments) + value(b, arguments)
        case Product(a, b):
            n = len(a.argument_spaces)
            return value(a, arguments[:n]) * value(b, arguments[n:])
        case Division(a, b):
            return value(a, arguments) / value(b)
        case Conjugate(a):
            return np.conj(value(a, tuple(_forms.conjugated(v) for v in arguments)))
        case RealPart(a):
            return np.real(value(a, arguments))
        case ImaginaryPart(a):
            return np.imag(value(a, arguments))
        case Absolute(a):
            return np.abs(value(a, arguments))
        case SquareRoot(a) | Logarithm(a) | Power(a, _):
            cast = complex if node.scalar_type == "complex" else lambda v: float(np.real(v))
            base = cast(value(a, arguments))
            if isinstance(node, Power):
                return np.power(base, cast(value(node.exponent)))
            return np.sqrt(base) if isinstance(node, SquareRoot) else np.log(base)
        case Determinant(a):
            return np.linalg.det(value(a))
        case Inverse(a):
            matrix = np.asarray(value(a))
            return (
                np.linalg.inv(matrix)
                if matrix.shape[0] == matrix.shape[1]
                else np.linalg.pinv(matrix)
            )
        case BroadcastTo(x, shape):
            target = tuple(_tensors.size(n, frame) for n in shape)
            if 0 in target:
                dtype = {"boolean": bool, "complex": complex, "real": float}[node.scalar_type]
                return np.empty(target, dtype=dtype)
            return np.broadcast_to(value(x, arguments), target)
        case Tensor(entries):
            return np.array([value(entry, arguments) for entry in entries])
        case Indexed(x, indices):
            key = _tensors.flat(_tensors.index_value(i, frame.indices) for i in indices)
            return np.asarray(value(x, arguments))[key]
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
        case ExternalOperator(operation, operands):
            inputs = tuple(_forms.supplied(m, m, frame) for m in operands)
            result = frame.environment[operation](*inputs)
            return (
                result(frame.context.points[node.domain])
                if node.domain is not None and callable(result)
                else result
            )
        case DirectionalDerivative(body, variables, point, direction):
            return _forms.directional(body, variables, point, direction, frame, arguments)
        case Coordinate(domain):
            return np.asarray(frame.context.points[domain], dtype=float)
        case Gradient(x, domain):
            return _spatial.gradient(x, domain, frame, arguments)
        case Divergence(x, domain):
            return np.trace(_spatial.gradient(x, domain, frame, arguments), axis1=-2, axis2=-1)
        case Curl(x, domain):
            return _spatial.curl(x, domain, frame, arguments)
        case Integral(x, domain):
            return integrate(domain, lambda: value(x, arguments), frame)
        case PullBack(x, mapping):
            return _spatial.pull_back(x, mapping, frame, arguments)
        case PushForward(x, mesh, law):
            return _spatial.push_forward(x, mesh, law, frame, arguments)
        case Conormal(side, conormals):
            return _spatial.conormal(side, conormals, frame)
        case RuntimeArray():
            return frame.input(node, node.label)
        case VertexCoordinates(topology, vertices):
            return elements.vertex_value(topology, vertices, frame.context.points)
        case _ if elements.is_basix_basis(node):
            return elements.basis_value(node, frame.context.points)
    raise TypeError(f"the reference interpreter has no evaluation rule for {type(node).__name__}")
