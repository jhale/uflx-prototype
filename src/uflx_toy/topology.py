from .nodes import StaticTopology, TabulatedTopology


def static_topology(cell_types, cells, sides):
    return StaticTopology(tuple(cell_types), cells, sides)


def tabulated_topology(label, cell_types, num_cells, sides, tables):
    return TabulatedTopology(label, tuple(cell_types), tuple(num_cells), sides, tables)
