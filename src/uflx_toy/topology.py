from .nodes import StaticTopology, TabulatedTopology


def static_topology(cell_types, cells, sides):
    # Expand grouped vertex input into a fixed local configuration.
    if cells and cells[0] and isinstance(cells[0][0], int):
        return StaticTopology(tuple(cell_types), tuple((tuple(c),) for c in cells), sides)
    types, rows, declarations = [], [], []
    for group, (K, group_cells) in enumerate(zip(cell_types, cells, strict=True)):
        for vertices in group_cells:
            types.append(K)
            rows.append((tuple(vertices),))
            chosen = []
            for name, parent, parent_group in sides[group] if sides else ():
                candidates = [j for j, cell in enumerate(parent.cells)
                              if set(vertices) <= set(cell[0])]
                rank = sum(U is parent for _, U, _ in chosen)
                slot = candidates[min(rank, len(candidates) - 1)]
                chosen.append((name, parent, slot))
            declarations.append(tuple(chosen))
    return StaticTopology(tuple(types), tuple(rows), tuple(declarations) if sides else ())


def tabulated_topology(label, cell_types, sides, positions):
    return TabulatedTopology(label, tuple(cell_types), sides, positions)
