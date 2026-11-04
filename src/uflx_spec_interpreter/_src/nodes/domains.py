from __future__ import annotations

from abc import abstractmethod
from typing import final

from .base import Base
from .topology import AbstractCellType


class AbstractDomain(Base):
    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @property
    @abstractmethod
    def ambient_domain(self) -> AbstractDomain: ...

    @property
    def ambient_coordinate_domain(self) -> AbstractCoordinateDomain:
        # The first coordinate domain among the ambient domains.
        return self.ambient_domain.ambient_coordinate_domain


class AbstractCoordinateDomain(AbstractDomain):
    @property
    def ambient_domain(self) -> AbstractDomain:
        return self

    @property
    def ambient_coordinate_domain(self) -> AbstractCoordinateDomain:
        return self


@final
class RD(AbstractCoordinateDomain):
    identifier = "uflx.RD"
    dimension: int  # pyright: ignore[reportIncompatibleMethodOverride]

    def _validate(self) -> None:
        from .validation import _require

        if type(self.dimension) is not int:
            raise TypeError("dimension must be an integer")
        _require(self.dimension >= 0, "dimension must be nonnegative")


class AbstractReferenceCell(AbstractCoordinateDomain, AbstractCellType):
    # A cell type that is also a coordinate domain, whose vertices the core does not read.
    pass
