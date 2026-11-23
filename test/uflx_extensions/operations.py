"""Example operations used by the extension and oracle tests."""

from uflx_extensions.external_operators import AbstractOperation, OutputType


class Energy(AbstractOperation):
    identifier = "test.Energy"

    def output_type(self, operands):
        if len(operands) != 2:
            raise ValueError("expected two operands")
        return OutputType((), "real", None)

    def derivative_action(self):
        return EnergyTangent()


class EnergyTangent(AbstractOperation):
    identifier = "test.EnergyTangent"

    def output_type(self, operands):
        if len(operands) != 4:
            raise ValueError("expected two points and two directions")
        return OutputType((), "real", None)
