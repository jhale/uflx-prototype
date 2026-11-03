from .names import derived_index


def cell_index(T, j):
    from .nodes import AbstractTopology

    if not isinstance(T, AbstractTopology) or type(j) is not int:
        raise TypeError("cell_index takes a topology and an integer")
    if not 0 <= j < len(T.cell_types):
        raise ValueError("invalid cell group")
    return derived_index("uflx.cell_index", T, j, 0)
