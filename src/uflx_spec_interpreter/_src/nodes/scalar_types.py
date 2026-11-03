from __future__ import annotations

from typing import final

from .base import Base


@final
class ScalarType(Base):
    # Symbolic scalar types obey the same immutable construction rules as other graph objects.
    identifier = "uflx.ScalarType"
    value: str


ScalarType.real = ScalarType("real")
ScalarType.complex = ScalarType("complex")
