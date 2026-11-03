from .nodes import GatheredIndex, Index


def index(label):
    return Index(label)


def gathered_index(table, indices):
    return GatheredIndex(table, indices)
