import numpy as np

from uflx import add, component_tensor
from uflx._src.algorithms.names import derived_indices
from uflx._src.algorithms.substitution import substitute
from uflx._src.nodes import Conjugate, Form, Zero
from uflx._src.tensors import identity

from .evaluation import evaluate
from .folding import fold
from .pipeline import compile_form
from .tabulation import coefficients


def assemble(form, degree):
    """Evaluate a tensor over the declared local argument coefficients."""
    tensor, _ = _local_tensor(form, degree)
    return tensor


def _local_tensor(form, degree):
    # Return a local tensor; gathering and scattering belong to the calling library.
    # A form is linear in each argument, so with the unit vectors in place of the coefficients
    # of argument p, read with a free index J_p, the form is its local tensor in J_p.
    # The arguments of a form whose expression is a form come first, as in its static type.
    e, arguments = compile_form(form, degree), ()
    while isinstance(e, Form):
        e, arguments = e.expression, e.arguments + arguments
    rows, extents, replacements = [], [], {}
    for p, entry in enumerate(arguments):
        v = entry.operand if isinstance(entry, Conjugate) else entry
        J, k = derived_indices("compiler.assembly", (p,), 2)
        replacements[coefficients(v)] = component_tensor(identity(v.space.num_dofs)[J, k], (k,))
        rows.append(J)
        extents.append((J, v.space.num_dofs))
    e = substitute(e, replacements)
    # Unused arguments still determine the tensor's axes, even for an identically zero form.
    e = add(e, Zero(e.value_shape, tuple(extents), e.argument_spaces, e.scalar_type))
    folded = fold(e)
    order = [i for i, _ in e.free_indices]

    def tensor(env, xp=np):
        data = evaluate(e, env, xp, folded)
        return xp.transpose(data, [order.index(J) for J in rows]) if rows else data

    return tensor, arguments


def assemble_bound(form, degree, bindings):
    """Gather bound fields, evaluate a local tensor, and scatter its bound arguments.

    ``bindings`` maps symbolic functions to DofMaps. Unbound functions keep their
    local coefficient layout. Environment keys match the compiler's input labels.
    """
    for field, mapping in bindings.items():
        if len(mapping.indices) != field.space.num_dofs:
            raise ValueError("a binding needs one index per local coefficient")
    local, arguments = _local_tensor(form, degree)
    arguments = tuple(a.operand if isinstance(a, Conjugate) else a for a in arguments)

    def evaluate(env):
        data = dict(env)
        for field, mapping in bindings.items():
            key = f"coefficients:{field.label}"
            if key in data:
                data[key] = mapping.gather(data[key])
        result = local(data)
        for axis, argument in enumerate(arguments):
            if argument in bindings:
                result = bindings[argument].scatter(result, axis)
        return result

    return evaluate
