import itertools

import basix
import numpy as np

from uflx_spec_interpreter._src.calculus import rotation_of_gradient, trace_of_gradient
from uflx_spec_interpreter._src.extensions.external_operators import ExternalOperator
from uflx_spec_interpreter._src.meshes import is_cell
from uflx_spec_interpreter._src.nodes import (
    Absolute,
    AbstractCoordinateDomain,
    AbstractExtent,
    AbstractTopology,
    Apply,
    Boolean,
    BroadcastTo,
    CellMap,
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
    GatheredIndex,
    Gradient,
    ImaginaryPart,
    Index,
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
    SideMap,
    SquareRoot,
    Sum,
    Tensor,
    Zero,
)
from uflx_spec_interpreter._src.nodes.arguments import argument_members
from uflx_toy.basix_nodes import BasixBasis, BasixCell, BasixSpace
from uflx_toy.meshes import shape_functions
from uflx_toy.nodes import RuntimeArray, VertexCoordinates

EPS = 1e-4
H = 1e-5
# The current point on each domain, read when a value is computed.
POINTS = {}


def quadrature(n=5):
    # A collapsed Gauss rule on the unit triangle, exact for polynomials of high degree.
    x, w = np.polynomial.legendre.leggauss(n)
    x, w = (x + 1) / 2, w / 2
    return [
        (np.array([a, b * (1 - a)]), wa * wb * (1 - a))
        for a, wa in zip(x, w)
        for b, wb in zip(x, w)
    ]


QUAD = quadrature()


def at(P, point, fn):
    saved = POINTS.get(P)
    POINTS[P] = point
    try:
        return fn()
    finally:
        if saved is None:
            POINTS.pop(P, None)
        else:
            POINTS[P] = saved


def jacobian_of(chart, X):
    return np.array([(chart(X + H * e) - chart(X - H * e)) / (2 * H) for e in np.eye(len(X))]).T


def index_value(i, ienv):
    # The value of an entry: an int, an Index from the environment, or a read of a table.
    if isinstance(i, Index):
        return ienv[i]
    if isinstance(i, GatheredIndex):
        values = i.table.values
        for j in i.indices:
            values = values[index_value(j, ienv)]
        return values
    return i


def transformed(ghat, law, J):
    # K^T on each covariant axis and J on each contravariant axis of the last axes, divided by
    # det J, or by the density for a cell of lower dimension, to the weight.
    K = np.linalg.pinv(J)
    value = np.asarray(ghat, dtype=complex)
    p = len(law.variances)
    for a, variance in enumerate(law.variances):
        M = K.T if variance == "covariant" else J
        value = np.moveaxis(
            np.tensordot(value, M, axes=([value.ndim - p + a], [1])), -1, value.ndim - p + a
        )
    k, n = J.shape[1], J.shape[0]
    scale = np.linalg.det(J) if k == n else np.sqrt(np.linalg.det(J.T @ J))
    return value / scale**law.weight


def lookup(env, node, label):
    # An input by its node, or by the label that a consumer of a kernel passes.
    return env[node] if node in env else env[label]


def side_reference(sm, env, ienv):
    # Resolve the source reference point, then apply the declared inclusion.
    source = sm.source
    i = POINTS.get(("piece", source), 0)
    j, A = sm.entries[i]
    point = POINTS[source]
    if isinstance(source, Mesh):
        chart = piece_map(source[i], {}, env)
        point = reference_point(chart, point, source.dimension)
    inside = evaluate(A.parameterisation, env, ienv)
    target = at(A.parameter_domain, point, lambda: np.real(np.asarray(inside(()))))
    return i, j, A, point, target


def on_cell(D, slot, fn):
    return at(("piece", D), slot, fn)


def basix_function(space, coefficients):
    # The value at the reference point of the current cell of the topology of the space.
    T = space.domain
    j = POINTS.get(("piece", T), 0)
    phi = space.element(j).tabulate(0, np.array([POINTS[T]], dtype=float))[0, 0]
    local = np.asarray(coefficients)[list(space.cell_dofs[j])]
    value = local.reshape(-1, space.block_size).T @ phi[:, 0] if space.block_shape else local @ phi
    return value.reshape(space.value_shape)


def piece_map(piece, members, env):
    # The map of one member of a piece, from reference coordinates to ambient coordinates.
    return lambda X: np.real(
        np.asarray(
            at(
                piece.reference_cell,
                np.asarray(X, dtype=float),
                lambda: evaluate(piece.parameterisation, env, members)(()),
            )
        )
    )


def reference_point(chart, x, dim):
    # Gauss-Newton for the reference point that the chart maps to x.
    if dim == 0:
        return np.empty(0)
    X = np.full(dim, 1 / (dim + 1))
    for _ in range(30):
        step = np.linalg.lstsq(jacobian_of(chart, X), chart(X) - np.real(x), rcond=None)[0]
        X = X - step
        if np.linalg.norm(step) < 1e-12:
            break
    return X


def tangent_projector(P, env):
    # The orthogonal projection onto the tangent space of P at the current point.
    if ("chart", P) in POINTS:
        T = jacobian_of(POINTS[("chart", P)], POINTS[("ref", P)])
    else:
        slot = POINTS.get(("piece", P))
        if slot is None:
            slot = locate(P, POINTS[P], env)
        piece = P[slot]
        chart = piece_map(piece, {}, env)
        T = jacobian_of(chart, reference_point(chart, POINTS[P], P.dimension))
    return T @ np.linalg.pinv(T)


def locate(P, x, env):
    # The cell of P that contains x, as a mesh library finds it for a point.
    for slot, piece in enumerate(P):
        chart = piece_map(piece, {}, env)
        X = reference_point(chart, x, P.dimension)
        if inside(piece.reference_cell, X) and np.linalg.norm(chart(X) - np.real(x)) < 1e-9:
            return slot
    raise ValueError(f"no cell of P contains {x}")


def rule_on(P):
    # Basix gives the rule on its cells, and the rules of the oracle are on the unit simplex.
    if isinstance(P, BasixCell):
        points, weights = basix.make_quadrature(P.cell_type, 12)
        return list(zip(points, weights, strict=True))
    if P.dimension == 0:
        return [(np.zeros(0), 1.0)]
    if P.dimension == 1:
        x, w = np.polynomial.legendre.leggauss(6)
        return [(np.array([(a + 1) / 2]), b / 2) for a, b in zip(x, w, strict=True)]
    return QUAD


def inside(K, X):
    # Whether X lies in the reference cell K, a unit simplex, square or cube.
    if isinstance(K, BasixCell) and K.cell_type in (
        basix.CellType.quadrilateral,
        basix.CellType.hexahedron,
    ):
        return np.all(X > -1e-9) and np.all(X < 1 + 1e-9)
    return np.all(X > -1e-9) and np.sum(X) < 1 + 1e-9


def reference_centroid(K):
    # The mean of the vertices of K, or of the unit simplex for a domain with no vertices.
    if hasattr(K, "vertices"):
        return np.mean(np.real(np.asarray(evaluate(K.vertices, {})(()))), axis=0)
    return np.full(K.dimension, 1 / (K.dimension + 1))


def integrate(P, fn, env):
    # Integrate each fixed local slot, counting coincident cells separately.
    if isinstance(P, AbstractCoordinateDomain):
        return sum(w * at(P, p, lambda: fn({})) for p, w in rule_on(P))
    total = 0
    for slot, piece in enumerate(P):
        psi = evaluate(piece.parameterisation, env)

        def mapped(inner, piece=piece, psi=psi, slot=slot):
            ref = POINTS[piece.reference_cell]
            chart = lambda X: np.real(np.asarray(at(piece.reference_cell, X, lambda: psi(()))))
            y = chart(ref)
            J = jacobian_of(chart, ref)
            scale = np.sqrt(abs(np.linalg.det(J.T @ J))) if len(ref) else 1.0
            value = lambda: on_cell(
                P, slot, lambda: at(P, y, lambda: at(P.ambient_domain, y, lambda: fn({})))
            )
            return at(("ref", P), ref, lambda: at(("chart", P), chart, value)) * scale

        total += integrate(piece.reference_cell, mapped, env)
    return total


# Closures for shared subexpressions are built once, since pass outputs are DAGs.
BUILT = {}
# Values at one state of POINTS.
VALUES = {}


def evaluate(e, env, ienv=None, over=None):
    ienv = {i: n for i, n in (ienv or {}).items() if i in dict(e.free_indices)}
    key = (e, id(env), frozenset(ienv.items()), id(over) if over else None)
    if key not in BUILT:
        f = evaluate_uncached(e, env, ienv, over)

        def shared(vals):
            if vals != ():
                return f(vals)
            at_points = (
                id(f),
                tuple(
                    (k, v.tobytes() if isinstance(v, np.ndarray) else v) for k, v in POINTS.items()
                ),
            )
            if at_points not in VALUES:
                if len(VALUES) > 10**6:
                    VALUES.clear()
                VALUES[at_points] = f(vals)
            return VALUES[at_points]

        BUILT[key] = (env, over, shared)
    return BUILT[key][-1]


def entry_value(member, value):
    # A value given to an entry conjugate(v) is the value of conjugate(v).
    if not isinstance(member, Conjugate):
        return member, value
    return member.operand, conjugated_value(value)


def complex_operation(op, fa):
    return lambda vals: op(np.asarray(fa(vals), dtype=complex))


# The value of an expression is a function of the values of its arguments.
def evaluate_uncached(e, env, ienv=None, over=None):
    ienv, over = ienv or {}, over or {}
    ev = lambda x, ie=ienv, ov=over: evaluate(x, env, ie, ov)

    def supplied(x, signature):
        if isinstance(x, tuple):
            return tuple(supplied(member, slot) for member, slot in zip(x, signature, strict=True))
        fx = ev(x)

        def bound_value(*slots):
            def read():
                value = fx(slots)
                return (
                    np.asarray(value, dtype=complex)
                    if signature.scalar_type == "complex"
                    else value
                )

            if signature.domain is not None:
                return lambda point: at(signature.domain, point, read)
            return read()

        return bound_value if x.argument_spaces else bound_value()

    if over and e in over:
        value = over[e]
        return lambda vals: value(*vals) if callable(value) else value
    match e:
        case Boolean(v) | Integer(v) | Float(v):
            return lambda vals: v
        case LessThan(a, b):
            return lambda vals: np.real(ev(a)(())) < np.real(ev(b)(()))
        case Conditional(predicate, left, right):

            def selected(vals):
                branch = left if ev(predicate)(()) else right
                value = ev(branch)(vals)
                return np.asarray(value, dtype=complex) if e.scalar_type == "complex" else value

            return selected
        case ExternalOperator(operation, operands):

            def external_value(vals):
                inputs = tuple(supplied(member, member) for member in operands)
                result = env[operation](*inputs)
                return (
                    result(POINTS[e.domain])
                    if e.domain is not None and callable(result)
                    else result
                )

            return external_value
        case Complex(a, b):
            return lambda vals: complex(a.value, b.value)
        case PushForward(g, D, law):
            fg = ev(g)

            def pushed(vals):
                # The value of g on the topology at the reference point of the point, by the law.
                j = POINTS.get(("piece", D), 0)
                piece = D[j]
                chart = piece_map(piece, {}, env)
                X = reference_point(chart, POINTS[D], D.dimension)
                T = D.topology
                ghat = on_cell(T, j, lambda: at(T, X, lambda: np.asarray(fg(vals))))
                return transformed(ghat, law, jacobian_of(chart, X))

            return pushed
        case VertexCoordinates(T, V):

            def vertex_value(vals):
                # The shape functions of the vertices of the current cell at the reference point.
                j = POINTS.get(("piece", T), 0)
                phi = shape_functions(T.cell_types[j], POINTS[T])
                return phi @ np.asarray(V)[list(T.cells[j])]

            return vertex_value
        case BasixBasis(space, slot):

            def basis_value(vals):
                j = POINTS.get(("piece", space.domain), 0)
                if j != slot:
                    return np.zeros(e.value_shape)
                phi = space.element(slot).tabulate(0, np.asarray([POINTS[space.domain]]))[0, 0]
                return phi.reshape(e.value_shape)

            return basis_value
        case Function(BasixSpace() as space, _):

            def basix_value(vals):
                # The coefficients of the dofs of the cell times the basis, as the library of V.
                return basix_function(space, lookup(env, e, f"coefficients:{e.label}"))

            return basix_value
        case Function(space, _) if space.argument_spaces:

            def applied(vals):
                result = env[e](*vals)
                return (
                    result(POINTS[space.domain])
                    if space.domain is not None and callable(result)
                    else result
                )

            return applied
        case Function(space, _) if space.domain is not None:

            def function_value(vals):
                v = env[e]
                if isinstance(v, dict):
                    # A function that differs between cells, read on the current cell.
                    D = space.domain
                    key = POINTS.get(("piece", D), 0)
                    return v[key](POINTS[D])
                return v(POINTS[space.domain]) if callable(v) else v

            return function_value
        case Function():
            return lambda vals: env[e]() if callable(env[e]) else env[e]
        case Zero(s, _, _, scalar_type):
            dtype = complex if scalar_type == "complex" else float
            return lambda vals: (
                np.zeros(tuple(size(n, env) for n in s), dtype=dtype) if s else dtype(0)
            )
        case Sum(a, b):
            fa, fb = ev(a), ev(b)
            return lambda vals: fa(vals) + fb(vals)
        case Product(a, b):
            fa, fb, n = ev(a), ev(b), len(a.argument_spaces)
            return lambda vals: fa(vals[:n]) * fb(vals[n:])
        case Division(a, c):
            fa, fc = ev(a), ev(c)
            return lambda vals: fa(vals) / fc(())
        case Conjugate(a):
            fa = ev(a)
            return lambda vals: np.conj(fa(tuple(conjugated_value(v) for v in vals)))
        case RealPart(a):
            return complex_operation(np.real, ev(a))
        case ImaginaryPart(a):
            return complex_operation(np.imag, ev(a))
        case Absolute(a):
            return complex_operation(np.abs, ev(a))
        case SquareRoot(a):
            fa = ev(a)
            return lambda vals: np.sqrt(
                complex(fa(vals)) if e.scalar_type == "complex" else np.real(fa(vals))
            )
        case Logarithm(a):
            fa = ev(a)
            return lambda vals: np.log(
                complex(fa(vals)) if e.scalar_type == "complex" else np.real(fa(vals))
            )
        case Determinant(a):
            fa = ev(a)
            return lambda vals: np.linalg.det(fa(vals))
        case Inverse(a):
            fa = ev(a)
            return lambda vals: np.linalg.pinv(np.asarray(fa(vals), dtype=complex))
        case Power(a, c):
            fa, fc = ev(a), ev(c)
            cast = complex if e.scalar_type == "complex" else lambda v: float(np.real(v))
            return lambda vals: np.power(cast(fa(vals)), cast(fc(())))
        case BroadcastTo(x, shape):
            target = tuple(size(n, env) for n in shape)
            if 0 in target:
                dtype = {"boolean": bool, "complex": complex, "real": float}[e.scalar_type]
                return lambda vals: np.empty(target, dtype=dtype)
            fx = ev(x)
            return lambda vals: np.broadcast_to(fx(vals), target)
        case Tensor(es):
            fs = [ev(x) for x in es]
            return lambda vals: np.array([f(vals) for f in fs])
        case Indexed(x, idx):
            fx = ev(x)

            # Every index selects one value axis.
            values = [index_value(i, ienv) for i in idx]
            key = tuple(k for v in values for k in argument_members(v))
            return lambda vals: np.asarray(fx(vals))[key]
        case ComponentTensor(x, idx) | IndexSum(x, idx):
            # Enumerate the bound axes independently.
            extents = [size(dict(x.free_indices)[i], env) for i in idx]
            ranges = [
                list(itertools.product(*map(range, n))) if isinstance(n, tuple) else range(n)
                for n in extents
            ]
            fs = {
                combo: ev(x, ienv | dict(zip(idx, combo))) for combo in itertools.product(*ranges)
            }
            shape = tuple(m for n in extents for m in argument_members(n))

            def flat(combo):
                return tuple(k for v in combo for k in argument_members(v))

            def value(vals):
                out = {combo: f(vals) for combo, f in fs.items()}
                if isinstance(e, IndexSum):
                    return sum(out.values())
                arr = np.empty(shape, dtype=bool if e.scalar_type == "boolean" else complex)
                for k, v in out.items():
                    arr[flat(k)] = v
                return arr

            return value
        case Form(x, args):
            n = len(x.argument_spaces)

            def value(vals):
                # The expression takes the first arguments, and the form binds the rest.
                bound = {}
                for a, val in zip(args, vals[n:]):
                    entries, values = (a, val) if isinstance(a, tuple) else ((a,), (val,))
                    bound |= dict(entry_value(m, v) for m, v in zip(entries, values, strict=True))
                # A bound value shadows an override of the same function.
                ov = {k: w for k, w in over.items() if k not in bound}
                return evaluate(x, env | bound, ienv, ov)(tuple(vals[:n]))

            return value
        case Apply(f, arguments):
            ff = ev(f)
            return lambda vals: ff(
                tuple(
                    supplied(arg, slot)
                    for arg, slot in zip(arguments, f.argument_spaces, strict=True)
                )
            )
        case DirectionalDerivative(f, x, a, h):
            # Every member of a tuple variable moves at once, along its member of the direction.
            xs = argument_members(x)
            fa, fh = [ev(m) for m in argument_members(a)], [ev(m) for m in argument_members(h)]

            def moved(step):
                return {
                    m: (lambda *slots, p=p, d=d: p(slots) + step * d(slots))
                    for m, p, d in zip(xs, fa, fh, strict=True)
                }

            fp = evaluate(f, env, ienv, over | moved(EPS))
            fm = evaluate(f, env, ienv, over | moved(-EPS))
            return lambda vals: (fp(vals) - fm(vals)) / (2 * EPS)
        case Coordinate(P):
            return lambda vals: np.asarray(POINTS[P], dtype=complex)
        case PullBack(b, SideMap() as sm):
            fb = ev(b)

            def restricted(vals):
                _, j, _, _, target = side_reference(sm, env, ienv)
                parent = sm.parent
                if isinstance(parent, Mesh):
                    target = piece_map(parent[j], {}, env)(target)
                return on_cell(parent, j, lambda: at(parent, target, lambda: fb(vals)))

            return restricted
        case PullBack(b, CellMap(AbstractTopology() as T, j) as m):
            fb = ev(b)

            def read(vals):
                # A point of a topology is a cell and a point of the reference cell of the cell.
                X = POINTS[m.reference_cell]
                return on_cell(T, j, lambda: at(T, X, lambda: fb(vals)))

            return read
        case PullBack(b, m):
            fb, psi = ev(b), ev(m.parameterisation)

            def pulled(vals):
                y = np.real(np.asarray(psi(())))
                if is_cell(m):
                    D = m.parent
                    return on_cell(D, m.cell_slot, lambda: at(D, y, lambda: fb(vals)))
                return at(m.ambient_domain, y, lambda: fb(vals))

            return pulled
        case Conormal(sm, conormals):

            def normal(vals):
                i, j, A, source_point, target = side_reference(sm, env, ienv)
                chart = piece_map(sm.parent[j], {}, env)
                J = jacobian_of(chart, target)
                field = evaluate(conormals[i], env, ienv)
                m = at(A.parameter_domain, source_point, lambda: np.real(np.asarray(field(()))))
                vector = np.linalg.pinv(J).T @ m
                return vector / np.linalg.norm(vector)

            return normal
        case Divergence(b, P):
            return ev(trace_of_gradient(b, P))
        case Curl(b, P):
            return ev(rotation_of_gradient(b, P))
        case Gradient(b, P):
            fb = ev(b)

            def gradient(vals):
                p = POINTS[P]
                cols = [
                    (
                        at(P, p + H * k, lambda: np.asarray(fb(vals)))
                        - at(P, p - H * k, lambda: np.asarray(fb(vals)))
                    )
                    / (2 * H)
                    for k in np.eye(len(p))
                ]
                g = np.stack(cols, axis=-1)
                if isinstance(P, Mesh) and P.dimension < len(p):
                    # The tangential gradient on a domain of lower dimension.
                    g = g @ tangent_projector(P, env)
                return g

            return gradient
        case Integral(b, P):
            if isinstance(P, AbstractCoordinateDomain):
                fb = ev(b)
                return lambda vals: integrate(P, lambda members: fb(vals), env)
            fs = {}

            def piece_value(members, vals):
                key = frozenset(members.items())
                if key not in fs:
                    fs[key] = evaluate(b, env, ienv, over)
                return fs[key](vals)

            return lambda vals: integrate(P, lambda members: piece_value(members, vals), env)
        case RuntimeArray():
            return lambda vals: lookup(env, e, e.label)
    raise TypeError(e)


def size(n, env):
    # The value of an AbstractExtent is bound in env, as a solver gives it to a kernel.
    if isinstance(n, tuple):
        return tuple(size(m, env) for m in n)
    return lookup(env, n, n.label) if isinstance(n, AbstractExtent) else n


def conjugated_value(value):
    if isinstance(value, tuple):
        return tuple(conjugated_value(v) for v in value)
    if callable(value):
        return lambda *args: conjugated_value(value(*(conjugated_value(arg) for arg in args)))
    return np.conj(value)
