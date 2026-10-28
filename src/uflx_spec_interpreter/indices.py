from .nodes import EllipsisIndex, GatheredIndex, Index, IndexTable


def index(label):
    return Index(label)


def ellipsis_index(label):
    return EllipsisIndex(label)


def index_table(values, extent):
    return IndexTable(values, extent)


def gathered_index(table, indices):
    return GatheredIndex(table, indices)
