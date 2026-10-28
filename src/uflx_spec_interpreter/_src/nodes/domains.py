from __future__ import annotations

from abc import abstractmethod
from functools import cached_property
from typing import final

from .base import AbstractExpression, Base
from .indices import Index
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
    def ambient_domain(self):
        return self

    @property
    def ambient_coordinate_domain(self):
        return self


@final
class RD(AbstractCoordinateDomain):
    identifier = "uflx.RD"
    dimension: int
    opaque = ("dimension",)


class AbstractReferenceCell(AbstractCoordinateDomain, AbstractCellType):
    # A cell type whose vertices have coordinates, row v of value shape (m, n) for vertex v.
    @property
    @abstractmethod
    def vertices(self) -> AbstractExpression: ...


@final
class DomainMap(Base):
    identifier = "uflx.DomainMap"
    parameter_domain: AbstractDomain
    parameterisation: AbstractExpression
    ambient_domain: AbstractDomain

    @cached_property
    def free_indices(self):
        return self.parameterisation.free_indices


@final
class DecomposedDomain(AbstractDomain):
    identifier = "uflx.DecomposedDomain"
    pieces: tuple[DomainMap, ...]

    @property
    def ambient_domain(self):
        return self.pieces[0].ambient_domain

    @property
    def dimension(self):
        return self.pieces[0].parameter_domain.dimension

    @cached_property
    def topology(self):
        # T if the free indices of piece j are (cell_index(T, j),) for every j, and otherwise None.
        found = set()
        for j, piece in enumerate(self.pieces):
            match piece.parameterisation.free_indices:
                case ((Index(("uflx.cell_index", T, jj, 0)), _),) if jj == j:
                    found.add(T)
                case _:
                    return None
        return found.pop() if len(found) == 1 else None


@final
class SideMap(Base):
    identifier = "uflx.SideMap"
    # The lazy map of the cells of the parameter domain into the ambient domain through a side.
    ambient_domain: DecomposedDomain
    parameter_domain: DecomposedDomain
    side: str
    opaque = ("side",)
    free_indices = ()
