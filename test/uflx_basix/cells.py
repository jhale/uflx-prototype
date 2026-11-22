"""Unit simplex fixture with explicit vertices and a Basix integration rule."""

from typing import final

import basix

from uflx.nodes import AbstractReferenceCell, Integer, ScalarType, Tensor, Zero


@final
class ReferenceSimplex(AbstractReferenceCell):
    identifier = "toy.ReferenceSimplex"
    # The unit simplex, with vertex 0 at the origin and vertex v the unit vector e_(v-1).
    _dimension: int

    @property
    def num_vertices(self):
        return self.dimension + 1

    @property
    def vertices(self):
        n = self.dimension
        if n == 0:
            return Zero((1, 0), (), (), ScalarType.real)
        return Tensor(
            tuple(
                Tensor(tuple(Integer(1 if k == v - 1 else 0) for k in range(n)))
                for v in range(n + 1)
            )
        )

    @property
    def dimension(self):
        return self._dimension

    def quadrature(self, degree):
        cells = (basix.CellType.interval, basix.CellType.triangle, basix.CellType.tetrahedron)
        if not 1 <= self.dimension <= len(cells):
            raise NotImplementedError(f"no simplex quadrature in dimension {self.dimension}")
        return basix.make_quadrature(cells[self.dimension - 1], degree)


def reference_simplex(n):
    return ReferenceSimplex(n)
