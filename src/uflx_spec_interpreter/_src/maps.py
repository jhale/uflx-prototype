from .nodes import DomainMap, PullBack


def domain_map(parameter_domain, parameterisation, ambient_domain):
    return DomainMap(parameter_domain, parameterisation, ambient_domain)


def pull_back(expression, domain):
    return PullBack(expression, domain)
