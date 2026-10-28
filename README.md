# UFLx spec interpreter

A toy implementation of the evaluation passes in the UFLx language specification
(`design/LANGUAGE.md` in the uflx repository), tested against an independent oracle.

The passes, in the order of the specification:

| Pass | Module | Output grammar |
| --- | --- | --- |
| 0 `apply_gateaux_derivatives` | `gateaux.py` | N1 |
| 1 `apply_actions` | `passes.py` | N2 |
| 2 `apply_derivatives` | `derivatives.py` | N3 |
| 3 `expand_integrals` | `integrals.py` | N4 |
| 4 `expand_pull_backs` | `pullbacks.py` | N5 |

`core.py` holds the node classes, substitution, freshening and the commuting
node classes. `static.py` computes static types, and `algebra.py` the matrix
compositions the passes share.

`oracle.py` evaluates an expression from the definitions of its nodes, with
finite differences for derivatives, quadrature for integrals and numeric maps
for pull-backs, so it does not use the passes.

Each `test_*.py` checks on random expressions that a pass keeps the value and the
static type, that its output is in its grammar, and that the grammar rejects its
inputs. `control_random.py` runs the negative controls, which must change values.

```
pip install -r requirements.txt
python test_full.py
./coverage_run.sh
```
