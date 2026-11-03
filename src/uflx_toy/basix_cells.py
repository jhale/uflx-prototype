from .basix_nodes import BasixCell


def basix_cell(cell_type):
    return BasixCell(cell_type.value)
