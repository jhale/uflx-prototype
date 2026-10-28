from .nodes import Index


def cell_index(T, j):
    return Index(("uflx.cell_index", T, j, 0))
