import itertools

import numpy as np

from uflx_spec_interpreter._src.nodes import (
    Absolute,
    AbstractCoordinateDomain,
    AbstractExtent,
    Action,
    CellMap,
    ComponentTensor,
    Conjugate,
    Coordinate,
    DecomposedDomain,
    DirectionalDerivative,
    Division,
    DualSpace,
    EllipsisIndex,
    Float,
    Form,
    Function,
    GateauxDerivative,
    GatheredIndex,
    Gradient,
    ImaginaryPart,
    ImaginaryUnit,
    Index,
    Indexed,
    IndexSum,
    Integer,
    Integral,
    Inverse,
    Logarithm,
    Normal,
    Power,
    Product,
    PullBack,
    RealPart,
    SideMap,
    SquareRoot,
    Sum,
    Tensor,
    Zero,
)
from uflx_spec_interpreter._src.static import arity, free, support
from uflx_toy.nodes import RuntimeArray

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
    if isinstance(i, Index | EllipsisIndex):
        return ienv[i]
    if isinstance(i, GatheredIndex):
        values = i.table.values
        for j in i.indices:
            values = values[index_value(j, ienv)]
        return values
    return i


def piece_map(piece, members):
    # The map of one member of a piece, from reference coordinates to ambient coordinates.
    return lambda X: np.real(
        np.asarray(
            at(
                piece.parameter_domain,
                np.asarray(X, dtype=float),
                lambda: evaluate(piece.parameterisation, {}, members)(()),
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


def tangent_projector(P):
    # The orthogonal projection onto the tangent space of P at the current point.
    if ("chart", P) in POINTS:
        T = jacobian_of(POINTS[("chart", P)], POINTS[("ref", P)])
    else:
        piece = P.pieces[0]
        index = next(iter(free(piece.parameterisation)))
        cell = POINTS[("cell", P)] if ("cell", P) in POINTS else locate(P, POINTS[P])
        chart = piece_map(piece, {index: cell})
        T = jacobian_of(chart, reference_point(chart, POINTS[P], P.dimension))
    return T @ np.linalg.pinv(T)


def locate(P, x):
    # The cell of P that contains x, as a mesh library finds it for a point.
    piece = P.pieces[0]
    index = next(iter(free(piece.parameterisation)))
    for cell in range(free(piece.parameterisation)[index]):
        chart = piece_map(piece, {index: cell})
        X = reference_point(chart, x, P.dimension)
        inside = np.all(X > -1e-9) and np.sum(X) < 1 + 1e-9
        if inside and np.linalg.norm(chart(X) - np.real(x)) < 1e-9:
            return cell
    raise ValueError(f"no cell of P contains {x}")


def integrate(P, fn):
    # The integral of fn over P, with fn reading the point of P from POINTS.
    if isinstance(P, AbstractCoordinateDomain):
        # The rules are on the unit simplex, so a coordinate domain of dimension 1 or more is one.
        if P.dimension == 0:
            rule = [(np.zeros(0), 1.0)]
        elif P.dimension == 1:
            x, w = np.polynomial.legendre.leggauss(6)
            rule = [(np.array([(a + 1) / 2]), b / 2) for a, b in zip(x, w)]
        else:
            rule = QUAD
        return sum(w * at(P, p, lambda: fn({})) for p, w in rule)
    total = 0
    for piece in P.pieces:
        fr = free(piece.parameterisation)
        for combo in itertools.product(*(range(n) for n in fr.values())):
            members = dict(zip(fr, combo))
            psi = evaluate(piece.parameterisation, {}, members)

            def mapped(inner, piece=piece, psi=psi, members=members):
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
                inside = lambda: at(P, y, lambda: at(P.ambient_domain, y, lambda: fn(members)))
                return (
                    at(
                        ("ref", P),
                        ref,
                        lambda: at(("chart", P), chart, lambda: at(("cell", P), cell, inside)),
                    )
                    * scale
                )

            total += integrate(piece.parameter_domain, mapped)
    return total


# Closures for shared subexpressions are built once, since pass outputs are DAGs.
BUILT = {}
# Values at one state of POINTS.
VALUES = {}


def evaluate(e, env, ienv=None, over=None):
    ienv = {i: n for i, n in (ienv or {}).items() if i in free(e)}
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
        case ImaginaryUnit():
            return lambda vals: 1j
        case Function(space, _) if space.domain is not None:

            def function_value(vals):
                v = env[e]
                if isinstance(v, dict):
                    # A function that differs between cells, read on the current cell.
                    return v[POINTS[("cell", space.domain)]](POINTS[space.domain])
                return v(POINTS[space.domain]) if callable(v) else v

            return function_value
        case Function(space, _) if not isinstance(space, DualSpace):
            return lambda vals: env[e]() if callable(env[e]) else env[e]
        case Function(space, _):
            if isinstance(space, DualSpace):
                return lambda vals: env[e](vals[0])
            return lambda vals: env[e]
        case Zero(s, _, _):
            return lambda vals: np.zeros(s) if s else 0.0
        case Sum(a, b):
            fa, fb = ev(a), ev(b)
            return lambda vals: fa(vals) + fb(vals)
        case Product(a, b):
            fa, fb, n = ev(a), ev(b), arity(a)
            return lambda vals: fa(vals[:n]) * fb(vals[n:])
        case Division(a, c):
            fa, fc = ev(a), ev(c)
            return lambda vals: fa(vals) / fc(())
        case Conjugate(a):
            fa = ev(a)
            return lambda vals: np.conj(fa(vals))
        case RealPart(a) | ImaginaryPart(a) | Absolute(a) | SquareRoot(a) | Logarithm(a):
            fa = ev(a)
            op = {
                RealPart: np.real,
                ImaginaryPart: np.imag,
                Absolute: np.abs,
                SquareRoot: np.sqrt,
                Logarithm: np.log,
            }[type(e)]
            return lambda vals: op(np.asarray(fa(vals), dtype=complex))
        case Inverse(a):
            fa = ev(a)
            return lambda vals: np.linalg.pinv(np.asarray(fa(vals), dtype=complex))
        case Power(a, c):
            fa, fc = ev(a), ev(c)
            return lambda vals: np.power(complex(fa(vals)), complex(fc(())))
        case Tensor(es):
            fs = [ev(x) for x in es]
            return lambda vals: np.array([f(vals) for f in fs])
        case Indexed(x, idx):
            fx = ev(x)

            # The value of an EllipsisIndex is a tuple, one entry for each axis that it covers.
            values = [index_value(i, ienv) for i in idx]
            key = tuple(k for v in values for k in (v if isinstance(v, tuple) else (v,)))
            return lambda vals: np.asarray(fx(vals))[key]
        case ComponentTensor(x, idx) | IndexSum(x, idx):
            # An EllipsisIndex ranges over the tuples of values of the axes that it covers.
            extents = [size(free(x)[i], env) for i in idx]
            ranges = [
                list(itertools.product(*map(range, n))) if isinstance(n, tuple) else range(n)
                for n in extents
            ]
            fs = {
                combo: ev(x, ienv | dict(zip(idx, combo))) for combo in itertools.product(*ranges)
            }
            shape = tuple(m for n in extents for m in (n if isinstance(n, tuple) else (n,)))

            def flat(combo):
                return tuple(k for v in combo for k in (v if isinstance(v, tuple) else (v,)))

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
            n = arity(x)

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
                return (lambda *point: fx(())) if support(x) is not None else fx(())

            gv = tuple(value_of(x) for x in g) if isinstance(g, tuple) else value_of(g)
            return lambda vals: ff((*tuple(vals[:p]), gv, *tuple(vals[p:])))
        case GateauxDerivative(F, x, a, entry):
            # The definition: the real derivative, split into its linear and antilinear parts.
            fa = ev(a)

            def delta(direction, rest):
                fp = evaluate(F, env, ienv, over | {x: lambda: fa(()) + EPS * direction})(rest)
                fm = evaluate(F, env, ienv, over | {x: lambda: fa(()) - EPS * direction})(rest)
                return (fp - fm) / (2 * EPS)

            def value(vals):
                rest, wv = tuple(vals[:-1]), vals[-1]
                # The value is that of the entry, so the direction of conjugate(w) is its conjugate.
                wv = np.conj(wv) if isinstance(entry, Conjugate) else wv
                if x.space.scalar_type == "real":
                    return delta(wv, rest)
                sign = 1j if isinstance(entry, Conjugate) else -1j
                return 0.5 * (delta(wv, rest) + sign * delta(1j * wv, rest))

            return value
        case DirectionalDerivative(f, x, a, h):
            fa, fh = ev(a), ev(h)
            fp = evaluate(f, env, ienv, over | {x: lambda: fa(()) + EPS * fh(())})
            fm = evaluate(f, env, ienv, over | {x: lambda: fa(()) - EPS * fh(())})
            return lambda vals: (fp(vals) - fm(vals)) / (2 * EPS)
        case Coordinate(P):
            return lambda vals: np.asarray(POINTS[P], dtype=complex)
        case PullBack(b, SideMap(Dd, Ed, cells, _)):
            fb = ev(b)

            def restricted(vals):
                # The point of E, read from the cell on the side through the table of cells.
                cell = cells.values[POINTS[("cell", Ed)]]
                return at(("cell", Dd), cell, lambda: at(Dd, POINTS[Ed], lambda: fb(vals)))

            return restricted
        case PullBack(b, m):
            fb, psi = ev(b), ev(m.parameterisation)

            def pulled(vals):
                y = np.real(np.asarray(psi(())))
                if isinstance(m, CellMap):
                    return at(
                        ("cell", m.ambient_domain),
                        index_value(m.cell_index, ienv),
                        lambda: at(m.ambient_domain, y, lambda: fb(vals)),
                    )
                return at(m.ambient_domain, y, lambda: fb(vals))

            return pulled
        case Normal(Dd, Ed, sidemap):
            cells = sidemap.cells

            def normal(vals):
                # In the tangent plane of the cell, orthogonal to the edge, away from the cell.
                x, e = POINTS[Ed], POINTS[("cell", Ed)]
                cell = cells.values[e]
                edge = piece_map(Ed.pieces[0], {next(iter(free(Ed.pieces[0].parameterisation))): e})
                t = jacobian_of(edge, reference_point(edge, x, 1))[:, 0]
                t = t / np.linalg.norm(t)
                chart = piece_map(
                    Dd.pieces[0], {next(iter(free(Dd.pieces[0].parameterisation))): cell}
                )
                X = reference_point(chart, x, Dd.dimension)
                d = jacobian_of(chart, X) @ (X - 1 / (len(X) + 1))
                n = d - (d @ t) * t
                return n / np.linalg.norm(n)

            return normal
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
                if isinstance(P, DecomposedDomain) and P.dimension < len(p):
                    # The tangential gradient on a domain of lower dimension.
                    g = g @ tangent_projector(P)
                return g

            return gradient
        case Integral(b, P):
            if isinstance(P, AbstractCoordinateDomain):
                fb = ev(b)
                return lambda vals: integrate(P, lambda members: fb(vals))
            fs = {}

            def piece_value(members, vals):
                key = frozenset(members.items())
                if key not in fs:
                    fs[key] = evaluate(b, env, ienv | members, over)
                return fs[key](vals)

            return lambda vals: integrate(P, lambda members: piece_value(members, vals))
        case RuntimeArray():
            return lambda vals: env[e]
    raise TypeError(e)


def size(n, env):
    # The value of an AbstractExtent is bound in env, as a solver gives it to a kernel.
    if isinstance(n, tuple):
        return tuple(size(m, env) for m in n)
    return env[n] if isinstance(n, AbstractExtent) else n
