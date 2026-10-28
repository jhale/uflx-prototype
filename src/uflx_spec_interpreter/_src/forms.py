from .nodes import Action, Conjugate, Form, Function, GateauxDerivative


def members(entry):
    return list(entry) if isinstance(entry, tuple) else [entry]


def entry_fns(entry):
    # The functions of an entry, without the conjugate of an entry conjugate(v).
    return [m.operand if isinstance(m, Conjugate) else m for m in members(entry)]


def function(space, label):
    return Function(space, label)


def form(expression, arguments):
    return Form(expression, tuple(arguments))


def action(form, position, argument):
    return Action(form, position, argument)


def gateaux_derivative(expression, variable, argument, point=None):
    # The point is the variable unless given, as for the free function of the specification.
    return GateauxDerivative(expression, variable, variable if point is None else point, argument)
