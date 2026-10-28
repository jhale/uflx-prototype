from .nodes import EllipsisIndex, GatheredIndex, Index


def index(label):
    return Index(label)


def ellipsis_index(label):
    return EllipsisIndex(label)


def gathered_index(table, indices):
    return GatheredIndex(table, indices)
