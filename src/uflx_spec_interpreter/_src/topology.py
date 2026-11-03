from .names import derived_index


def cell_index(T, j):
    return derived_index("uflx.cell_index", T, j, 0)
