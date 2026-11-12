from .nodes import StaticTopology, TabulatedTopology


def static_topology(cell_types, cells, sides):
    return StaticTopology(tuple(cell_types), tuple(cells), sides)


def tabulated_topology(label, cell_types, sides, positions):
    return TabulatedTopology(label, tuple(cell_types), sides, positions)
