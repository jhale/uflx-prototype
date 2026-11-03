from .nodes import (
    AbstractTopology,
    CellMap,
    GatheredIndex,
    Mesh,
    Normal,
    PullBack,
    PushForward,
    SideMap,
    TransformationLaw,
)
from .substitution import substitute
from .topology import cell_index


def cell_domain(D, j, e=None):
    # Private compatibility spelling; both topology and mesh cells are CellMaps.
    return cell_map(D, j, e)


def cell_map(X, j, e=None):
    T = X.topology if isinstance(X, Mesh) else X
    return CellMap(X, j, cell_index(T, j) if e is None else e)


def image(x, Q):
    return Mesh(x, Q)


def coordinate_function(D):
    return D.coordinate_function


IDENTITY = TransformationLaw((), 0)


def transformation_law(variances, weight):
    return TransformationLaw(variances, weight)


def push_forward(f, D, m):
    return PushForward(f, D, m)


def cell_of(m):
    return (
        (m.cell_group, m.entry)
        if isinstance(m, CellMap) and isinstance(m.ambient_domain, Mesh)
        else None
    )


def is_cell(m):
    return cell_of(m) is not None


def side_map(X, P, s):
    return SideMap(X, P, s)


def side_domain(D, S, s):
    x = coordinate_function(D)
    return image(PullBack(x, side_map(x.domain, S, s)), D.ambient_domain)


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
