from .nodes import Action, Conjugate, Form, Function, GateauxDerivative


def members(entry):
    return list(entry) if isinstance(entry, tuple) else [entry]


def entry_fns(entry):
    # The functions of an entry, without the conjugate of an entry conjugate(v).
    return [m.operand if isinstance(m, Conjugate) else m for m in members(entry)]


def function(V, label):
    return Function(V, label)


def form(expression, arguments):
    return Form(expression, tuple(arguments))


def action(f, position, g):
    return Action(f, position, g)


def gateaux_derivative(F, x, w, a=None):
    return GateauxDerivative(F, x, x if a is None else a, w)
