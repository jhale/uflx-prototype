# UFLx spec interpreter

A toy implementation of the evaluation passes in the UFLx language specification
(`design/LANGUAGE.md` in the uflx repository), tested against an independent oracle.

The passes, in `src/uflx_spec_interpreter/`, in the order of the specification:

| Pass | Module | Output grammar |
| --- | --- | --- |
| 0 `apply_gateaux_derivatives` | `passes/gateaux.py` | N1 |
| 1 `apply_actions` | `passes/actions.py` | N2 |
| 2 `apply_derivatives` | `passes/derivatives.py` | N3 |
| 3 `expand_integrals` | `passes/integrals.py` | N4 |
| 4 `expand_pull_backs` | `passes/pullbacks.py` | N5 |
| 5 `apply_gradients` | `passes/gradients.py` | N6 |

The package splits a public API from a private one.

The public API is the free functions, named as in the specification, in the
modules at the top of the package. A user, and a test that builds expressions,
imports only these:

- `literals.py`: `num`, which builds the literal of an `int` or a `float`, and a
  complex number as a composition with `ImaginaryUnit`, and `zero`
- `arithmetic.py`: `add`, `subtract`, `negate`, `multiply`, `divide`, `power`,
  the elementary functions and `total`, where `add`, `subtract` and `negate`
  work on tensors componentwise as the specification says
- `indices.py`: `index`, `index_table` and `gathered_index`
- `tensors.py`: `tensor`, `indexed`, `component_tensor`, `index_sum`, `inverse`,
  and the matrix compositions such as `transpose`, `matrix_product` and
  `determinant`
- `calculus.py`: `coordinate`, `gradient`, `directional_derivative` and
  `integral`
- `maps.py`: `domain_map` and `pull_back`
- `geometry.py`: `jacobian`, `inverse_jacobian`, `density` and the metric
- `domains.py`: `coordinate_domain`, `decomposed_domain`, `cell_domain`,
  `gathered_cell_domain`, `side_map` and `normal`
- `forms.py`: `space`, `function`, `form`, `action` and `gateaux_derivative`
- `static.py`: the free functions that read the static type

The private API is the classes in `nodes/`, which only the passes, the oracle
and the white-box tests import:

- `nodes/base.py`: `Base`, which reads the construction arguments of each class
  from its annotations, as rules 4 and 5 of the specification describe, and
  interns every object with a seed-free hash before it is built (see
  `INTERNING_PLAN.md`), and `AbstractExpression`, whose static type each
  concrete class states, with no defaults (see `ABSTRACT_EXPRESSION_PLAN.md`).
  Every concrete class is `@final`, and the abstract classes are contracts.
- `nodes/literals.py`, `nodes/arithmetic.py`, `nodes/indices.py`,
  `nodes/tensors.py`, `nodes/calculus.py`, `nodes/domains.py` and
  `nodes/forms.py`: the node classes, by section of the specification, each
  with a public module of the same name for its free functions
- `interning.py`: the table of nodes, their canonical text and the seed-free
  digest
- `substitution.py`: traversal, rebuilding, freshening and simultaneous
  substitution
- `passes/commuting.py`: the commuting node classes, into which an evaluated
  node moves
- `passes/derivative_table.py`: the derivatives of the node classes, shared by
  passes 2 and 5

`oracle.py` evaluates an expression from the definitions of its nodes, with
finite differences for derivatives, quadrature for integrals and numeric maps
for pull-backs, so it does not use the passes.

The tests in `test/` check on random expressions that a pass keeps the value
and the static type, that its output is in its grammar, and that the grammar
rejects its inputs. `test_controls.py` breaks the passes on purpose and checks
that the tests notice. Each random test also runs as a script, with the number
of trials as its argument, and prints its counts.

```
pip install -e ".[test,lint]"
pytest
ruff check . && ruff format --check .
coverage run -m pytest && coverage report
python test/test_full.py 100
```

## Differences from the specification

The classes, their fields, the passes and the helpers use the names of the
specification. The toy simplifies the specification in these places:

- `Space` is every space, with the flags `dual` and `complex`, where the
  specification has `AbstractFunctionSpace`, `DualSpace` and `ConjugateSpace`.
- `CoordinateDomain` is both `RD` and a reference cell, and has no vertices.
- `CellMap` is a cell of a decomposed domain, or a gathered cell, which the
  specification writes as a `DomainMap` through `cell_domain` and
  `gathered_cell_domain`.
- `arity` counts the arguments, where the specification has
  `argument_spaces`, so a `Zero` has placeholder argument spaces.
- `IndexTable` is an `AbstractIndexValue` whose values the toy knows, so the
  oracle can read them, where the specification leaves the values to a library.
- A `SideMap` holds its side tables, where the specification reads them from
  the topology by the name of the side.
- The constructors do not validate their arguments (rule 8).
