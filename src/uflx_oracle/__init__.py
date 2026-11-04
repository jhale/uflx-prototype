import itertools

import basix
import numpy as np

from uflx_spec_interpreter import cell_index
from uflx_spec_interpreter._src.calculus import _rotation_of_gradient, _trace_of_gradient
from uflx_spec_interpreter._src.meshes import _cell_of
from uflx_spec_interpreter._src.nodes import (
    Absolute,
    AbstractCoordinateDomain,
    AbstractExtent,
    AbstractTopology,
    Action,
    BroadcastTo,
    CellMap,
    Complex,
    ComponentTensor,
    Conjugate,
    Coordinate,
    Curl,
    Determinant,
    DirectionalDerivative,
    Divergence,
    Division,
    DualSpace,
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
    Logarithm,
    Mesh,
    Normal,
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
from uflx_spec_interpreter._src.tuples import _argument_members
from uflx_toy.basix_nodes import BasixCell, BasixSpace
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


def cell_value(m, ienv):
    # The group and the cell that a cell reads, as a library.
    found = _cell_of(m)
    return None if found is None else (found[0], index_value(found[1], ienv))


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


def side_cell(sm):
    # The group and the cell on the side of the current cell of the parameter domain of the side
    # map, through the side tables.
    E, S = sm.parameter_domain, sm.side_topology
    i = POINTS.get(("piece", E), 0)
    (_, _, j) = next(s for s in S.sides[i] if s[0] == sm.side)
    return j, S.side_tables(i, sm.side)[0].values[POINTS[("cell", E)]]


def on_cell(D, piece, cell, fn):
    return at(("piece", D), piece, lambda: at(("cell", D), cell, fn))


def basix_function(space, coefficients):
    # The value at the reference point of the current cell of the topology of the space.
    T = space.domain
    j, cell = POINTS.get(("piece", T), 0), POINTS[("cell", T)]
    phi = space.element(j).tabulate(0, np.array([POINTS[T]], dtype=float))[0, 0]
    local = np.asarray(coefficients)[list(space.dofmap[j][cell])]
    value = local.reshape(-1, space.block_size).T @ phi[:, 0] if space.block_shape else local @ phi
    return value.reshape(space.value_shape)


def piece_map(piece, members, env):
    # The map of one member of a piece, from reference coordinates to ambient coordinates.
    return lambda X: np.real(
        np.asarray(
            at(
                piece.parameter_domain,
                np.asarray(X, dtype=float),
                lambda: evaluate(piece.parameterisation, env, members)(()),
            )
        )
    )


def reference_point(chart, x, dim):
    # Gauss-Newton for the reference point that the chart maps to x.
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
        piece = P.pieces[0]
        index = next(iter(dict(piece.parameterisation.free_indices)))
        cell = POINTS[("cell", P)] if ("cell", P) in POINTS else locate(P, POINTS[P], env)
        chart = piece_map(piece, {index: cell}, env)
        T = jacobian_of(chart, reference_point(chart, POINTS[P], P.dimension))
    return T @ np.linalg.pinv(T)


def locate(P, x, env):
    # The cell of P that contains x, as a mesh library finds it for a point.
    piece = P.pieces[0]
    index = next(iter(dict(piece.parameterisation.free_indices)))
    for cell in range(dict(piece.parameterisation.free_indices)[index]):
        chart = piece_map(piece, {index: cell}, env)
        X = reference_point(chart, x, P.dimension)
        if inside(piece.parameter_domain, X) and np.linalg.norm(chart(X) - np.real(x)) < 1e-9:
            return cell
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
    # The integral of fn over P, with fn reading the point of P from POINTS.
    if isinstance(P, AbstractCoordinateDomain):
        return sum(w * at(P, p, lambda: fn({})) for p, w in rule_on(P))
    total = 0
    for number, piece in enumerate(P.pieces):
        fr = dict(piece.parameterisation.free_indices)
        for combo in itertools.product(*(range(n) for n in fr.values())):
            members = dict(zip(fr, combo))
            psi = evaluate(piece.parameterisation, env, members)

            def mapped(inner, piece=piece, psi=psi, members=members, number=number):
                y = np.real(np.asarray(psi(())))
                if isinstance(piece.parameter_domain, AbstractCoordinateDomain):
                    ref = POINTS[piece.parameter_domain]
                    chart_in = lambda X: X
                else:
                    ref, chart_in = (
                        POINTS[("ref", piece.parameter_domain)],
                        POINTS[("chart", piece.parameter_domain)],
                    )
                chart = lambda X: np.real(
                    np.asarray(at(piece.parameter_domain, chart_in(X), lambda: psi(())))
                )
                if len(ref) == 0:
                    scale = 1.0
                else:
                    T = jacobian_of(chart_in, ref)
                    Je = jacobian_of(chart, ref)
                    scale = np.sqrt(abs(np.linalg.det(Je.T @ Je)) / abs(np.linalg.det(T.T @ T)))
                cell = next(iter(members.values()), None)
                inside = lambda: at(
                    ("piece", P),
                    number,
                    lambda: at(P, y, lambda: at(P.ambient_domain, y, lambda: fn(members))),
                )
                return (
                    at(
                        ("ref", P),
                        ref,
                        lambda: at(("chart", P), chart, lambda: at(("cell", P), cell, inside)),
                    )
                    * scale
                )

            total += integrate(piece.parameter_domain, mapped, env)
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
    if callable(value):
        return member.operand, lambda *point: np.conj(value(*point))
    return member.operand, np.conj(value)


def complex_operation(op, fa):
    return lambda vals: op(np.asarray(fa(vals), dtype=complex))


# The value of an expression is a function of the values of its arguments.
def evaluate_uncached(e, env, ienv=None, over=None):
    ienv, over = ienv or {}, over or {}
    ev = lambda x, ie=ienv, ov=over: evaluate(x, env, ie, ov)
    if over and e in over:
        value = over[e]
        return lambda vals: value() if callable(value) else value
    match e:
        case Integer(v) | Float(v):
            return lambda vals: v
        case Complex(a, b):
            return lambda vals: complex(a.value, b.value)
        case PushForward(g, D, law):
            fg = ev(g)

            def pushed(vals):
                # The value of g on the topology at the reference point of the point, by the law.
                j, cell = POINTS.get(("piece", D), 0), POINTS[("cell", D)]
                piece = D.pieces[j]
                chart = piece_map(
                    piece, {next(iter(dict(piece.parameterisation.free_indices))): cell}, env
                )
                X = reference_point(chart, POINTS[D], D.dimension)
                T = D.topology
                ghat = on_cell(T, j, cell, lambda: at(T, X, lambda: np.asarray(fg(vals))))
                return transformed(ghat, law, jacobian_of(chart, X))

            return pushed
        case VertexCoordinates(T, V):

            def vertex_value(vals):
                # The shape functions of the vertices of the current cell at the reference point.
                j, cell = POINTS.get(("piece", T), 0), POINTS[("cell", T)]
                phi = shape_functions(T.cell_types[j], POINTS[T])
                return phi @ np.asarray(V)[list(T.cells[j][cell])]

            return vertex_value
        case Function(BasixSpace() as space, _):

            def basix_value(vals):
                # The coefficients of the dofs of the cell times the basis, as the library of V.
                return basix_function(space, lookup(env, e, f"coefficients:{e.label}"))

            return basix_value
        case Function(space, _) if space.domain is not None:

            def function_value(vals):
                v = env[e]
                if isinstance(v, dict):
                    # A function that differs between cells, read on the current cell.
                    D = space.domain
                    cell = POINTS[("cell", D)]
                    key = (POINTS.get(("piece", D), 0), cell) if len(D.pieces) > 1 else cell
                    return v[key](POINTS[D])
                return v(POINTS[space.domain]) if callable(v) else v

            return function_value
        case Function(space, _) if not isinstance(space, DualSpace):
            return lambda vals: env[e]() if callable(env[e]) else env[e]
        case Function(space, _):
            if isinstance(space, DualSpace):
                return lambda vals: env[e](vals[0])
            return lambda vals: env[e]
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
            return complex_operation(np.sqrt, ev(a))
        case Logarithm(a):
            return complex_operation(np.log, ev(a))
        case Determinant(a):
            fa = ev(a)
            return lambda vals: np.linalg.det(fa(vals))
        case Inverse(a):
            fa = ev(a)
            return lambda vals: np.linalg.pinv(np.asarray(fa(vals), dtype=complex))
        case Power(a, c):
            fa, fc = ev(a), ev(c)
            return lambda vals: np.power(complex(fa(vals)), complex(fc(())))
        case BroadcastTo(x, shape):
            target = tuple(size(n, env) for n in shape)
            if 0 in target:
                dtype = complex if e.scalar_type == "complex" else float
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
            key = tuple(k for v in values for k in _argument_members(v))
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
            shape = tuple(m for n in extents for m in _argument_members(n))

            def flat(combo):
                return tuple(k for v in combo for k in _argument_members(v))

            def value(vals):
                out = {combo: f(vals) for combo, f in fs.items()}
                if isinstance(e, IndexSum):
                    return sum(out.values())
                arr = np.empty(shape, dtype=complex)
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
        case Action(f, p, g):
            ff = ev(f)

            def value_of(x):
                # A field is bound lazily, as a function of the point, and anything else eagerly.
                fx = ev(x)
                return (lambda *point: fx(())) if x.domain is not None else fx(())

            gv = tuple(value_of(x) for x in g) if isinstance(g, tuple) else value_of(g)
            return lambda vals: ff((*tuple(vals[:p]), gv, *tuple(vals[p:])))
        case DirectionalDerivative(f, x, a, h):
            # Every member of a tuple variable moves at once, along its member of the direction.
            xs = _argument_members(x)
            fa, fh = [ev(m) for m in _argument_members(a)], [ev(m) for m in _argument_members(h)]

            def moved(step):
                return {
                    m: (lambda p=p, d=d: p(()) + step * d(()))
                    for m, p, d in zip(xs, fa, fh, strict=True)
                }

            fp = evaluate(f, env, ienv, over | moved(EPS))
            fm = evaluate(f, env, ienv, over | moved(-EPS))
            return lambda vals: (fp(vals) - fm(vals)) / (2 * EPS)
        case Coordinate(P):
            return lambda vals: np.asarray(POINTS[P], dtype=complex)
        case PullBack(b, SideMap(AbstractTopology() as T, S, side) as sm):
            fb = ev(b)

            def restricted_on_topology(vals):
                # The reference point of the cell on the side, through the inclusion A of the cell.
                i, e = POINTS.get(("piece", S), 0), POINTS[("cell", S)]
                _, A, _ = S.side_tables(i, side)
                inside = evaluate(A.parameterisation, env, {cell_index(S, i): e})
                X = at(A.parameter_domain, POINTS[S], lambda: np.real(np.asarray(inside(()))))
                j, cell = side_cell(sm)
                return on_cell(T, j, cell, lambda: at(T, X, lambda: fb(vals)))

            return restricted_on_topology
        case PullBack(b, SideMap(Dd) as sm):
            fb, Ed = ev(b), sm.parameter_domain

            def restricted(vals):
                # The point of E, read from the cell on the side through the table of cells.
                j, cell = side_cell(sm)
                return on_cell(Dd, j, cell, lambda: at(Dd, POINTS[Ed], lambda: fb(vals)))

            return restricted
        case PullBack(b, CellMap(AbstractTopology() as T, j, entry) as m):
            fb = ev(b)

            def read(vals):
                # A point of a topology is a cell and a point of the reference cell of the cell.
                X = POINTS[m.parameter_domain]
                cell = index_value(entry, ienv)
                return on_cell(T, j, cell, lambda: at(T, X, lambda: fb(vals)))

            return read
        case PullBack(b, m):
            fb, psi = ev(b), ev(m.parameterisation)

            def pulled(vals):
                y = np.real(np.asarray(psi(())))
                found = cell_value(m, ienv)
                if found is not None:
                    D = m.ambient_domain
                    return on_cell(D, *found, lambda: at(D, y, lambda: fb(vals)))
                return at(m.ambient_domain, y, lambda: fb(vals))

            return pulled
        case Normal(SideMap(Dd) as sm):
            Ed = sm.parameter_domain

            def normal(vals):
                # In the tangent plane of the cell, orthogonal to the edge, away from the cell.
                x, e = POINTS[Ed], POINTS[("cell", Ed)]
                j, cell = side_cell(sm)
                facet = Ed.pieces[POINTS.get(("piece", Ed), 0)]
                edge = piece_map(
                    facet, {next(iter(dict(facet.parameterisation.free_indices))): e}, env
                )
                t = jacobian_of(edge, reference_point(edge, x, 1))[:, 0]
                t = t / np.linalg.norm(t)
                piece = Dd.pieces[j]
                chart = piece_map(
                    piece, {next(iter(dict(piece.parameterisation.free_indices))): cell}, env
                )
                X = reference_point(chart, x, Dd.dimension)
                d = jacobian_of(chart, X) @ (X - reference_centroid(piece.parameter_domain))
                n = d - (d @ t) * t
                return n / np.linalg.norm(n)

            return normal
        case Divergence(b, P):
            return ev(_trace_of_gradient(b, P))
        case Curl(b, P):
            return ev(_rotation_of_gradient(b, P))
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
        return lambda *args: np.conj(value(*args))
    return np.conj(value)
