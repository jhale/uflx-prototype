"""An abstract field space for testing the UFLx language."""

from uflx.nodes import AbstractDomain, AbstractFunctionSpace, AbstractTopology, ScalarType


class FieldSpace(AbstractFunctionSpace):
    identifier = "test.FieldSpace"
    name: str
    _value_shape: tuple
    support: AbstractDomain | AbstractTopology | tuple
    _scalar_type: str
    argument_spaces = ()

    @property
    def domain(self):
        return self.support or None

    @property
    def value_shape(self):
        return self._value_shape

    @property
    def scalar_type(self):
        return self._scalar_type


def space(name, value_shape=(), domain=None, scalar_type=ScalarType.complex):
    return FieldSpace(name, tuple(value_shape), () if domain is None else domain, scalar_type)
