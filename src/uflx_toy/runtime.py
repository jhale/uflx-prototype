from .nodes import RuntimeArray, RuntimeExtent


def runtime_extent(label):
    return RuntimeExtent(label)


def runtime_array(label, value_shape):
    return RuntimeArray(label, tuple(value_shape))
