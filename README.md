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

The language itself is split by concern:

- `nodes.py`: the node classes, from the expressions to the form layer, interned
  with a seed-free hash so that equal nodes are the same object (see
  `INTERNING_PLAN.md`)
- `domains.py`: domains, domain maps and their cells
- `substitution.py`: traversal, freshening and simultaneous substitution
- `commuting.py`: the commuting node classes, into which an evaluated node moves
- `calculus.py`: the derivatives of the node classes, shared by passes 2 and 5
- `geometry.py`: the Jacobian matrix, its inverse on the tangent space, and the density
- `static.py`: static types
- `algebra.py`: the matrix compositions the passes share

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
