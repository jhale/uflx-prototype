from .nodes import DomainMap, PullBack


def domain_map(P, psi, Q):
    return DomainMap(P, psi, Q)


def pull_back(f, D):
    return PullBack(f, D)
