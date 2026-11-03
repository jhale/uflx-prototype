import numpy as np

from uflx_spec_interpreter import component_tensor
from uflx_spec_interpreter._src.names import derived_indices
from uflx_spec_interpreter._src.nodes import Conjugate, Form
from uflx_spec_interpreter._src.substitution import substitute
from uflx_spec_interpreter._src.tensors import identity

from .evaluation import evaluate
from .folding import fold
from .pipeline import compile_form
from .tabulation import coefficients


def assemble(form, degree):
    # A form is linear in each argument, so with the unit vectors in place of the coefficients
    # of argument p, read with a free index J_p, the form is its assembled tensor in J_p.
    # The arguments of a form whose expression is a form come first, as in its static type.
    e, arguments = compile_form(form, degree), ()
    while isinstance(e, Form):
        e, arguments = e.expression, e.arguments + arguments
    rows, replacements = [], {}
    for p, entry in enumerate(arguments):
        v = entry.operand if isinstance(entry, Conjugate) else entry
        J, k = derived_indices("compiler.assembly", (p,), 2)
        replacements[coefficients(v)] = component_tensor(identity(v.space.num_dofs)[J, k], (k,))
        rows.append(J)
    e = substitute(e, replacements)
    folded = fold(e)
    order = [i for i, _ in e.free_indices]

    def tensor(env, xp=np):
        data = evaluate(e, env, xp, folded)
        return xp.transpose(data, [order.index(J) for J in rows]) if rows else data

    return tensor
