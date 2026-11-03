from uflx_spec_interpreter import ScalarType

from .basix_nodes import BasixSpace


def basix_space(label, domain, family, degree, discontinuous=False, scalar_type=ScalarType.real):
    return BasixSpace(label, domain, family.value, degree, int(discontinuous), scalar_type)
