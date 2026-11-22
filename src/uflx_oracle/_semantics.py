"""Numerical semantics of the expression nodes, evaluated with lexical bindings."""

from dataclasses import dataclass, field

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

from . import _dual as ad
from . import _finite_elements as elements
from . import _forms, _spatial, _tensors
from ._context import Operand
from ._external import evaluate_external
from ._geometry import integrate


@dataclass(frozen=True, eq=False)
class Frame:
    """Evaluate with fixed function bindings, index assignments and spatial points."""

    environment: dict
    indices: dict
    points: dict
    cells: dict
    _memo: dict = field(default_factory=dict, repr=False)

    def value(self, node, arguments=()):
        if arguments:
            return interpret(node, self, arguments)
        # Only the node's free indices can affect its value at this point.
        key = (node, tuple(self.indices.get(i) for i, _ in node.free_indices))
        if key not in self._memo:
            self._memo[key] = interpret(node, self, ())
        return self._memo[key]

    def at(self, points, cells=None):
        return Frame(
            self.environment,
            self.indices,
            self.points | points,
            self.cells if cells is None else self.cells | cells,
        )

    def sampled_in(self, caller):
        """Keep lexical bindings and use the points and cells at the sampling site."""
        if self.points is caller.points and self.cells is caller.cells:
            return self
        return Frame(self.environment, self.indices, caller.points, caller.cells)

    def for_comparison(self):
        # A predicate reads values at the current point, including the point of a
        # directional derivative. Neither its operands nor its outcome need tangents.
        if not any(
            isinstance(v, (ad.Dual, Operand)) for v in self.environment.values()
        ) and not any(isinstance(p, ad.Dual) for p in self.points.values()):
            return self

        def value_only(binding):
            if isinstance(binding, Operand):
                return type(binding)(lambda *args: value_only(binding(*args)))
            return ad.primal(binding)

        return Frame(
            {node: value_only(v) for node, v in self.environment.items()},
            self.indices,
            {domain: ad.primal(p) for domain, p in self.points.items()},
            self.cells,
        )

    def bind_arguments(self, bindings):
        return Frame(self.environment | bindings, self.indices, self.points, self.cells)

    def bind_indices(self, indices):
        return Frame(self.environment, self.indices | indices, self.points, self.cells, self._memo)

    def input(self, node, label):
        """Library data may be supplied by node or by its runtime label."""
        return self.environment[node] if node in self.environment else self.environment[label]


def interpret(node, frame, arguments):
    binding = frame.environment.get(node)
    if isinstance(binding, _forms.ExpressionBinding):
        return binding.read(frame, arguments)

    value = frame.value
    match node:
        case Boolean(v) | Integer(v) | Float(v):
            return v
        case Complex(a, b):
            return complex(a.value, b.value)
        case LessThan(a, b):
            local = frame.for_comparison()
            return np.real(local.value(a)) < np.real(local.value(b))
        case Conditional(predicate, left, right):
            result = value(left if value(predicate) else right, arguments)
            return ad.asarray(result, dtype=complex) if node.scalar_type == "complex" else result
        case Zero(shape, _, _, scalar_type):
            dtype = complex if scalar_type == "complex" else float
            return (
                np.zeros(tuple(_tensors.size(n, frame) for n in shape), dtype=dtype)
                if shape
                else dtype(0)
            )
        case Sum(a, b):
            return ad.add(value(a, arguments), value(b, arguments))
        case Product(a, b):
            n = len(a.argument_spaces)
            return ad.multiply(value(a, arguments[:n]), value(b, arguments[n:]))
        case Division(a, b):
            return ad.divide(value(a, arguments), value(b))
        case Conjugate(a):
            return ad.linear(np.conj, value(a, tuple(_forms.conjugated(v) for v in arguments)))
        case RealPart(a):
            return ad.linear(np.real, value(a, arguments))
        case ImaginaryPart(a):
            return ad.linear(np.imag, value(a, arguments))
        case Absolute(a):
            return ad.absolute(value(a, arguments))
        case SquareRoot(a) | Logarithm(a) | Power(a, _):

            def cast(v):
                if node.scalar_type == "complex":
                    return ad.asarray(v, dtype=complex)
                return ad.asarray(ad.linear(np.real, v), dtype=float)

            base = cast(value(a, arguments))
            if isinstance(node, Power):
                return ad.power(base, cast(value(node.exponent)))
            return ad.sqrt(base) if isinstance(node, SquareRoot) else ad.logarithm(base)
        case Determinant(a):
            return ad.determinant(value(a))
        case Inverse(a):
            matrix = ad.asarray(value(a))
            rows, columns = np.shape(ad.primal(matrix))
            return ad.inverse(matrix) if rows == columns else ad.pinv(matrix)
        case BroadcastTo(x, shape):
            target = tuple(_tensors.size(n, frame) for n in shape)
            if 0 in target:
                dtype = {"boolean": bool, "complex": complex, "real": float}[node.scalar_type]
                return np.empty(target, dtype=dtype)
            return ad.linear(lambda a: np.broadcast_to(a, target), value(x, arguments))
        case Tensor(entries):
            return ad.stack([value(entry, arguments) for entry in entries])
        case Indexed(x, indices):
            key = _tensors.flat(_tensors.index_value(i, frame.indices) for i in indices)
            return ad.entry(ad.asarray(value(x, arguments)), key)
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
            return ad.asarray(frame.points[domain], dtype=float)
        case Gradient(x, domain):
            return _spatial.gradient(x, domain, frame, arguments)
        case Divergence(x, domain):
            return ad.linear(
                lambda g: np.trace(g, axis1=-2, axis2=-1),
                _spatial.gradient(x, domain, frame, arguments),
            )
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
            return frame.input(node, node.label)
        case VertexCoordinates(topology, vertices):
            return elements.vertex_value(topology, vertices, frame)
        case _ if elements.is_basix_basis(node):
            return elements.basis_value(node, frame)
    raise TypeError(f"the reference interpreter has no evaluation rule for {type(node).__name__}")
