"""Explicit library-side DOF bindings for the small assembled examples."""

from dataclasses import dataclass

import basix
import numpy as np


@dataclass(frozen=True)
class DofMap:
    """Map each packed local coefficient to an assembled DOF; repetitions are allowed."""

    indices: tuple[int, ...]
    num_dofs: int

    def __post_init__(self):
        if self.num_dofs < 0 or any(i < 0 or i >= self.num_dofs for i in self.indices):
            raise ValueError("DOF indices must lie inside the assembled vector")

    def gather(self, values, xp=np):
        """Pack assembled coefficients in the local expression's cell order."""
        values = xp.asarray(values)
        if values.shape[0] != self.num_dofs:
            raise ValueError("wrong assembled coefficient count")
        return xp.take(values, xp.asarray(self.indices, dtype=int), axis=0)

    def scatter(self, values, axis=0):
        """Add a local tensor axis into assembled DOFs, including repeated entries."""
        values = np.moveaxis(np.asarray(values), axis, 0)
        if values.shape[0] != len(self.indices):
            raise ValueError("wrong local coefficient count")
        result = np.zeros((self.num_dofs, *values.shape[1:]), dtype=values.dtype)
        np.add.at(result, np.asarray(self.indices, dtype=int), values)
        return np.moveaxis(result, 0, axis)


def conforming_dofmap(space):
    """Identify entity DOFs from vertex labels, for elements needing no transformations.

    This is an explicit toy assembly policy, not part of BasixSpace or the language.
    A solver supplies its own maps and DOF transformations for general elements.
    """
    T, first, count, rows = space.topology, {}, 0, []
    for j, vertices in enumerate(T.cells):
        E = space.element(j)
        if not E.dof_transformations_are_identity:
            raise NotImplementedError("toy conforming DOF maps do not apply DOF transformations")
        entities = basix.cell.topology(T.cell_types[j].cell_type)
        tdim = len(entities) - 1
        row = [0] * E.dim
        for d, on_dim in enumerate(E.entity_dofs):
            for n, local in enumerate(on_dim):
                key = (
                    ("cell", j)
                    if d == tdim
                    else (d, frozenset(vertices[v] for v in entities[d][n]))
                )
                if key not in first:
                    first[key], count = count, count + len(local)
                for m, dof in enumerate(local):
                    row[dof] = first[key] + m
        rows.extend(space.block_size * dof + c for dof in row for c in range(space.block_size))
    return DofMap(tuple(rows), count * space.block_size)
