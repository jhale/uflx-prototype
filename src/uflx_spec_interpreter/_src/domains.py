from .nodes import (
    AbstractTopology,
    CellMap,
    Coordinate,
    DecomposedDomain,
    DomainMap,
    GatheredIndex,
    Normal,
    PullBack,
    PushForward,
    SideMap,
    TransformationLaw,
)
from .substitution import substitute
from .topology import cell_index


def cell_domain(D, j, e=None):
    # Piece j with the ambient domain D, and for a mesh cell e of group j through its cell map.
    piece = D.pieces[j]
    psi = piece.parameterisation
    if e is not None:
        x, c = psi.expression, psi.domain
        psi = PullBack(x, CellMap(c.topology, c.group, e))
    return DomainMap(piece.parameter_domain, psi, D)


def cell_domain_of_specification(D, j):
    # The signature of LANGUAGE.md, which the draft of coordinate functions extends with e.
    return cell_domain(D, j)


def cell_map(T, j, e=None):
    return CellMap(T, j, cell_index(T, j) if e is None else e)


def mesh(x, Q):
    T = x.support
    pieces = (DomainMap(K, PullBack(x, cell_map(T, j)), Q) for j, K in enumerate(T.cell_types))
    return DecomposedDomain(tuple(pieces))


def coordinate_function(D):
    # The expression on a topology that a mesh is built from.
    return D.pieces[0].parameterisation.expression


IDENTITY = TransformationLaw((), 0)


def transformation_law(variances, weight):
    return TransformationLaw(variances, weight)


def push_forward(f, D, m=None):
    # By the law of the space of a function, or by the law m. The coordinate function of D by
    # the identity is coordinate(D), equal in value, which the gradient of a coordinate reads.
    m = f.space.law if m is None else m
    if m == IDENTITY and f == coordinate_function(D):
        return Coordinate(D)
    return PushForward(f, D, m)


def decomposed_domain(pieces):
    return DecomposedDomain(tuple(pieces))


def gathered_cell_domain(D, j, g):
    return cell_domain(D, j, g)


def cell_of(m):
    # The group and the entry of a cell of a mesh, read from its cell map.
    match m:
        case DomainMap(_, PullBack(_, CellMap(_, j, e)), D) if m == cell_domain(D, j, e):
            return j, e
    return None


def is_cell(m):
    return cell_of(m) is not None


def side_map(X, P, s=None):
    # To the side topology P from a topology, or to a side domain P from a mesh, through the side
    # s of every group, or through the first side of each group on the topology of X.
    T = X if isinstance(X, AbstractTopology) else X.topology
    S = P if isinstance(P, AbstractTopology) else P.topology
    names = tuple(s or next(n for n, U, _ in sides if U == T) for sides in S.sides)
    return SideMap(X, P, names)


def side_map_of_specification(D, E, s):
    # The signature of LANGUAGE.md, which the draft extends with a topology for D and E.
    return side_map(D, E, s)


def side_domain(D, S):
    # The mesh of the trace of the coordinate function of D through the first side of each group.
    x = coordinate_function(D)
    return mesh(PullBack(x, side_map(x.support, S)), D.ambient_domain)


def inclusion(X, S, i, s, e):
    # B, the cell on the side s of cell e of group i, and A, the inclusion of cell e into it.
    (_, _, j) = next(side for side in S.sides[i] if side[0] == s)
    cells, A, _ = S.side_tables(i, s)
    c = cell_index(S, i)
    g = GatheredIndex(cells, (e,))
    B = CellMap(X, j, g) if isinstance(X, AbstractTopology) else cell_domain(X, j, g)
    return B, substitute(A, {c: e})


def normal(sm):
    return Normal(sm)
