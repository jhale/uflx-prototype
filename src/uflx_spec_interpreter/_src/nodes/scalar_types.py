from enum import Enum


class ScalarType(Enum):
    # The arithmetic in which an expression is evaluated, with real embedded in complex.
    real = "real"
    complex = "complex"
