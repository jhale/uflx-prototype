from .nodes import Conjugate


def members(entry):
    return list(entry) if isinstance(entry, tuple) else [entry]


def entry_fns(entry):
    # The functions of an entry, without the conjugate of an entry conjugate(v).
    return [m.operand if isinstance(m, Conjugate) else m for m in members(entry)]
