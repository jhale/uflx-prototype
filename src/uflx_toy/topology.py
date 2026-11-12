from .nodes import StaticTopology, TabulatedTopology


def static_topology(cell_types, cells):
    return StaticTopology(tuple(cell_types), tuple(cells))


def tabulated_topology(label, cell_types):
    return TabulatedTopology(label, tuple(cell_types))
