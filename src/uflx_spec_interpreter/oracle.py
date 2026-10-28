import itertools

import numpy as np

from .core import CellMap, Gathered, Normal, SideMap, Gateaux, Inv, CT, D, Abs, Action, Conj, Div, Im, Log, Power, Re, Sqrt, Fn, Form, Index, Indexed, ISum, Num, Prod, Sum, Tensor, Zero, entry_fns, members
from .static import arity, free, support
from .core import CoordDomain, Coordinate, Decomposed, Gradient, Integral, PullBack, ambient

EPS = 1e-4
H = 1e-5
# The current point on each domain, read when a value is computed.
POINTS = {}


def quadrature(n=5):
    # A collapsed Gauss rule on the unit triangle, exact for polynomials of high degree.
    x, w = np.polynomial.legendre.leggauss(n)
    x, w = (x + 1) / 2, w / 2
    return [(np.array([a, b * (1 - a)]), wa * wb * (1 - a)) for a, wa in zip(x, w) for b, wb in zip(x, w)]


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


def topdim(P):
    return P.dim if isinstance(P, CoordDomain) else topdim(P.pieces[0].P)


def tangent_projector(P):
    # The orthogonal projection onto the tangent space of P at the current point.
    if ("chart", P) in POINTS:
        T = jacobian_of(POINTS[("chart", P)], POINTS[("ref", P)])
    else:
        piece = P.pieces[0]
        cell = POINTS[("cell", P)]
        index = list(free(piece.psi))[0]
        verts = [np.real(np.asarray(at(piece.P, np.array(v, dtype=float), lambda: evaluate(piece.psi, {}, {index: cell})(())))) for v in ((0, 0), (1, 0), (0, 1))]
        T = np.array([verts[1] - verts[0], verts[2] - verts[0]]).T
    return T @ np.linalg.pinv(T)


def integrate(P, fn):
    # The integral of fn over P, with fn reading the point of P from POINTS.
    if isinstance(P, CoordDomain):
        if P.dim == 0:
            rule = [(np.zeros(0), 1.0)]
        elif P.dim == 1:
            x, w = np.polynomial.legendre.leggauss(6)
            rule = [(np.array([(a + 1) / 2]), b / 2) for a, b in zip(x, w)]
        else:
            rule = QUAD
        return sum(w * at(P, p, lambda: fn({})) for p, w in rule)
    total = 0
    for piece in P.pieces:
        fr = free(piece.psi)
        for combo in itertools.product(*(range(n) for n in fr.values())):
            members = dict(zip(fr, combo))
            psi = evaluate(piece.psi, {}, members)

            def mapped(inner, piece=piece, psi=psi, members=members):
                y = np.real(np.asarray(psi(())))
                if isinstance(piece.P, CoordDomain):
                    ref = POINTS[piece.P]
                    chart_in = lambda X: X
                else:
                    ref, chart_in = POINTS[("ref", piece.P)], POINTS[("chart", piece.P)]
                chart = lambda X: np.real(np.asarray(at(piece.P, chart_in(X), lambda: psi(()))))
                if len(ref) == 0:
                    scale = 1.0
                else:
                    T = jacobian_of(chart_in, ref)
                    Je = jacobian_of(chart, ref)
                    scale = np.sqrt(abs(np.linalg.det(Je.T @ Je)) / abs(np.linalg.det(T.T @ T)))
                cell = next(iter(members.values()), None)
                inside = lambda: at(P, y, lambda: at(ambient(P), y, lambda: fn(members)))
                return at(("ref", P), ref, lambda: at(("chart", P), chart, lambda: at(("cell", P), cell, inside))) * scale

            total += integrate(piece.P, mapped)
    return total


# Closures for shared subexpressions are built once, since pass outputs are DAGs.
BUILT = {}
# Values at one state of POINTS.
VALUES = {}


def evaluate(e, env, ienv=None, over=None):
    ienv = {i: n for i, n in (ienv or {}).items() if i in free(e)}
    key = (id(e), id(env), frozenset(ienv.items()), id(over) if over else None)
    if key not in BUILT:
        f = evaluate_uncached(e, env, ienv, over)

        def shared(vals):
            if vals != ():
                return f(vals)
            at_points = (id(f), tuple((id(k), v.tobytes() if isinstance(v, np.ndarray) else v) for k, v in POINTS.items()))
            if at_points not in VALUES:
                if len(VALUES) > 10**6:
                    VALUES.clear()
                VALUES[at_points] = f(vals)
            return VALUES[at_points]

        BUILT[key] = (e, env, over, shared)
    return BUILT[key][-1]


# The value of an expression is a function of the values of its arguments.
def evaluate_uncached(e, env, ienv=None, over=None):
    ienv, over = ienv or {}, over or {}
    ev = lambda x, ie=ienv, ov=over: evaluate(x, env, ie, ov)
    if over and e in over:
        value = over[e]
        return lambda vals: value() if callable(value) else value
    match e:
        case Num(v):
            return lambda vals: v
        case Fn(space, _) if space.domain is not None:

            def function_value(vals):
                v = env[e]
                if isinstance(v, dict):
                    # A function that differs between cells, read on the current cell.
                    return v[POINTS[("cell", space.domain)]](POINTS[space.domain])
                return v(POINTS[space.domain]) if callable(v) else v

            return function_value
        case Fn(space, _) if not space.dual:
            return lambda vals: env[e]() if callable(env[e]) else env[e]
        case Fn(space, _):
            if space.dual:
                return lambda vals: env[e](vals[0])
            return lambda vals: env[e]
        case Zero(s, _, _):
            return lambda vals: np.zeros(s) if s else 0.0
        case Sum(a, b):
            fa, fb = ev(a), ev(b)
            return lambda vals: fa(vals) + fb(vals)
        case Prod(a, b):
            fa, fb, n = ev(a), ev(b), arity(a)
            return lambda vals: fa(vals[:n]) * fb(vals[n:])
        case Div(a, c):
            fa, fc = ev(a), ev(c)
            return lambda vals: fa(vals) / fc(())
        case Conj(a):
            fa = ev(a)
            return lambda vals: np.conj(fa(vals))
        case Re(a) | Im(a) | Abs(a) | Sqrt(a) | Log(a):
            fa = ev(a)
            op = {Re: np.real, Im: np.imag, Abs: np.abs, Sqrt: np.sqrt, Log: np.log}[type(e)]
            return lambda vals: op(np.asarray(fa(vals), dtype=complex))
        case Inv(a):
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
            def entry(i):
                if isinstance(i, Index):
                    return ienv[i]
                if isinstance(i, Gathered):
                    table = np.asarray(evaluate(i.table, env, ienv, over)(()))
                    return int(np.real(table[tuple(entry(j) for j in i.indices)]))
                return i

            key = tuple(entry(i) for i in idx)
            return lambda vals: np.asarray(fx(vals))[key]
        case CT(x, idx) | ISum(x, idx):
            extents = [free(x)[i] for i in idx]
            fs = {combo: ev(x, ienv | dict(zip(idx, combo))) for combo in itertools.product(*(range(n) for n in extents))}

            def value(vals):
                out = {combo: f(vals) for combo, f in fs.items()}
                if isinstance(e, ISum):
                    return sum(out.values())
                arr = np.empty(extents, dtype=complex)
                for k, v in out.items():
                    arr[k] = v
                return arr

            return value
        case Form(x, args):

            n = arity(x)

            def value(vals):
                # The expression takes the first arguments, and the form binds the rest.
                bound = {}
                for a, val in zip(args, vals[n:]):
                    names = entry_fns(a)
                    bound |= dict(zip(names, val if isinstance(a, tuple) else (val,)))
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
            return lambda vals: ff(tuple(vals[:p]) + (gv,) + tuple(vals[p:]))
        case Gateaux(F, x, a, entry):
            # The definition: the real derivative, split into its linear and antilinear parts.
            fa = ev(a)

            def delta(direction, rest):
                fp = evaluate(F, env, ienv, over | {x: lambda: fa(()) + EPS * direction})(rest)
                fm = evaluate(F, env, ienv, over | {x: lambda: fa(()) - EPS * direction})(rest)
                return (fp - fm) / (2 * EPS)

            def value(vals):
                rest, wv = tuple(vals[:-1]), vals[-1]
                if not x.space.complex:
                    return delta(wv, rest)
                sign = 1j if isinstance(entry, Conj) else -1j
                return 0.5 * (delta(wv, rest) + sign * delta(1j * wv, rest))

            return value
        case D(f, x, a, h):
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
                cell = int(np.real(np.asarray(evaluate(cells, env)(()))[POINTS[("cell", Ed)]]))
                return at(("cell", Dd), cell, lambda: at(Dd, POINTS[Ed], lambda: fb(vals)))

            return restricted
        case PullBack(b, m):
            fb, psi = ev(b), ev(m.psi)

            def pulled(vals):
                y = np.real(np.asarray(psi(())))
                if isinstance(m, CellMap):
                    cell = evaluate(Indexed(Tensor((Num(0), Num(1), Num(2), Num(3))), (m.entry,)), env, ienv, over)(())
                    return at(("cell", m.Q), int(np.real(cell)), lambda: at(m.Q, y, lambda: fb(vals)))
                return at(m.Q, y, lambda: fb(vals))

            return pulled
        case Normal(Dd, Ed, sidemap):
            cells = sidemap.cells

            def normal(vals):
                # The outward normal of the cell on the side, from the geometry alone.
                x, e = POINTS[Ed], POINTS[("cell", Ed)]
                cell = int(np.real(np.asarray(evaluate(cells, env)(()))[e]))
                piece = Ed.pieces[0]
                ends = [np.real(np.asarray(at(piece.P, np.array([s]), lambda: evaluate(piece.psi, env, {list(free(piece.psi))[0]: e})(())))) for s in (0.0, 1.0)]
                tangent = (ends[1] - ends[0]) / np.linalg.norm(ends[1] - ends[0])
                cpiece = Dd.pieces[0]
                cidx = list(free(cpiece.psi))[0]
                verts = [np.real(np.asarray(at(cpiece.P, np.array(v, dtype=float), lambda: evaluate(cpiece.psi, env, {cidx: cell})(())))) for v in ((0, 0), (1, 0), (0, 1))]
                # In the plane of the cell, orthogonal to the edge, away from the centroid of the cell.
                d = np.real(x) - sum(verts) / 3
                n = d - (d @ tangent) * tangent
                return n / np.linalg.norm(n)

            return normal
        case Gradient(b, P):
            fb = ev(b)

            def gradient(vals):
                p = POINTS[P]
                cols = [(at(P, p + H * k, lambda: np.asarray(fb(vals))) - at(P, p - H * k, lambda: np.asarray(fb(vals)))) / (2 * H) for k in np.eye(len(p))]
                g = np.stack(cols, axis=-1)
                if isinstance(P, Decomposed) and topdim(P) < len(p):
                    # The tangential gradient on a domain of lower dimension.
                    g = g @ tangent_projector(P)
                return g

            return gradient
        case Integral(b, P):
            if isinstance(P, CoordDomain):
                fb = ev(b)
                return lambda vals: integrate(P, lambda members: fb(vals))
            fs = {}

            def piece_value(members, vals):
                key = frozenset(members.items())
                if key not in fs:
                    fs[key] = evaluate(b, env, ienv | members, over)
                return fs[key](vals)

            return lambda vals: integrate(P, lambda members: piece_value(members, vals))
    raise TypeError(e)
