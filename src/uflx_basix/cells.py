from .nodes import BasixCell


def cell(cell_type):
    return BasixCell(cell_type.value)
