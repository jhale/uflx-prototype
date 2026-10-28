from uflx_spec_interpreter._src.nodes import AbstractExtent
from uflx_spec_interpreter._src.substitution import expressions
from uflx_toy.nodes import RuntimeArray


def kernel_inputs(e):
    # The arrays that a consumer passes by label, with the label of an extent known only at
    # assembly in place of its length, which the consumer passes under that label too. The
    # basis tables are left out, because runtime_tables computes them from the other inputs.
    found, seen = {}, set()

    def walk(n):
        if n in seen:
            return
        seen.add(n)
        basis = isinstance(n, RuntimeArray) and isinstance(n.label, tuple)
        if basis and n.label[0] == "basis":
            # The points of a basis table are inputs, from which runtime_tables computes it.
            walk(n.label[3])
        elif isinstance(n, RuntimeArray):
            shape = n.value_shape
            found[n.label] = tuple(m.label if isinstance(m, AbstractExtent) else m for m in shape)
        for c in expressions(n):
            walk(c)

    walk(e)
    return found
