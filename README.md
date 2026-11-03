# UFLx spec interpreter

`lower(f)` runs the standard core recipe and checks its core-normal-form output.
Individual transformations remain available. Internal checks in `_src/contracts.py`
return offending nodes and diagnostic reasons without running transformations or
checking later compiler capabilities.

`gateaux_derivative(...)` constructs a `Form` from `DirectionalDerivative`
expressions. There is no separate Gateaux node or pass. Linearity analysis is an
optional internal optimization, not a requirement for declaring a form.

`image(x, Q)` constructs a `Mesh` from topology-supported coordinates with no
free indices. Mesh integrals retain free integrand indices and sum over internal
cell indices. Cell-dependent data are fields on the topology or mesh; use mapped
reference-domain integrals and explicit index sums for other integration families.


A toy implementation of the evaluation passes in the UFLx language specification
(`design/LANGUAGE.md` in the uflx repository), tested against an independent oracle.

The repository has four packages in `src/`, and imports go only from the oracle
and the compiler to the extension to the core, which `test/test_boundaries.py`
checks:

- `uflx_spec_interpreter`: the core, with only what the specification defines
- `uflx_toy`: a language extension, with the classes that the specification
  leaves to a library or to an extension, `IndexTable`, `RuntimeExtent`,
  `RuntimeArray`, the space `Space`, the reference cell `ReferenceSimplex`, and the
  topologies `StaticTopology` and `TabulatedTopology`, whose side tables
  `inclusions.py` builds from vertex positions with `affine_map`, and `BasixCell`,
  a reference cell of Basix, whose facet normals the side tables read from the
  tables of Basix, and `BasixSpace`, a space of Basix elements on a
  mesh, whose dof map shares the dofs of shared entities so that the space is
  continuous
- `uflx_oracle`: the oracle that the tests compare the passes against
- `uflx_compiler`: a form compiler after the core normal form, with a quadrature
  pass that rewrites an integral over a Basix cell as a weighted sum over the
  points of a rule of Basix, a tabulation pass that rewrites a function of a
  `BasixSpace` at the points as its coefficients times the tabulated basis, a
  table that `runtime_tables` fills at assembly for points known only then,
  `kernel_inputs`, which lists the arrays and extents that a consumer passes by
  label, such as `"coefficients:u"`, and
  an evaluator of the arithmetic that remains on NumPy or JAX arrays, after
  `fold` has computed once the subgraphs that read no kernel input
  and `assemble`, which puts the unit vectors in place of the coefficients of
  each argument of a form, so that the form evaluates to its assembled tensor

The standard recipe in `_src/passes/lowering.py` coordinates:

| Transformation | Module |
| --- | --- |
| Apply actions | `passes/actions.py` |
| Expand derivatives and finish resulting actions | `passes/derivatives.py` |
| Expand mesh integrals | `passes/integrals.py` |
| Alternate gradients and pull-backs until stable | `passes/gradients.py`, `passes/pullbacks.py` |
| Check core normal form | `contracts.py` |

The shape extension uses its own recipe: expose geometry, apply shape derivatives,
then re-enter checked core lowering. Quadrature and tabulation follow that boundary
in the compiler and impose their own requirements.

The core splits a public API from a private one, as JAX does with `jax._src`.

The public API is `uflx_spec_interpreter/__init__.py`, which re-exports the
free functions of the specification and declares them in `__all__`, with the
stand-in `coordinate_domain` and the short name `num`. A user,
and a test that builds expressions, imports only these, as in
`from uflx_spec_interpreter import add, index`. `AbstractExpression` also has
the operator overloads of the specification, which call these free functions
and reject Python numbers.

Everything else is in the private package `_src`, which only the toy
extension, the oracle and the white-box tests import. The toy imports only
`_src.nodes`, for the base classes it extends. The free functions are in the
modules of `_src`:

- `literals.py`: `num`, which builds the literal of an `int` or a `float`, and a
  complex number as a composition with `ImaginaryUnit`, and `zero`
- `arithmetic.py`: `add`, `subtract`, `negate`, `multiply`, `divide`, `power`,
  the elementary functions and `total`, where `add`, `multiply`, `divide`,
  `real_part` and `imaginary_part` read the ranks and build the compositions of
  tensor arithmetic as the specification says
- `indices.py`: `index`, `ellipsis_index` and `gathered_index`
- `tensors.py`: `tensor`, `indexed`, `component`, `component_tensor`,
  `index_sum`, `inverse`, and the matrix compositions such as `transpose`,
  `matrix_product` and `determinant`
- `calculus.py`: `coordinate`, `gradient`, `directional_derivative` and
  `integral`
- `maps.py`: `domain_map` and `pull_back`
- `geometry.py`: `jacobian`, `inverse_jacobian`, `density` and the metric
- `topology.py`: `cell_index`
- `domains.py`: `image`, `cell_map`, `cell_domain`, `side_domain`,
  `side_map` and `normal`
- `spaces.py`: `conjugate_space` and `dual_space`
- `forms.py`: `function`, `form`, `action` and `gateaux_derivative`
- `static.py`: the free functions that read the static type

The private modules of `_src`:

- `nodes/base.py`: `Base`, which reads the construction arguments of each class
  from its annotations, as rules 4 and 5 of the specification describe, and
  interns every object with a seed-free hash before it is built (see
  `INTERNING_PLAN.md`), and `AbstractExpression`, whose static type each
  concrete class states, with no defaults (see `ABSTRACT_EXPRESSION_PLAN.md`).
  Every concrete class is `@final`, and the abstract classes are contracts.
- `nodes/literals.py`, `nodes/arithmetic.py`, `nodes/indices.py`,
  `nodes/tensors.py`, `nodes/calculus.py`, `nodes/domains.py`,
  `nodes/topology.py`, `nodes/spaces.py` and `nodes/forms.py`: the node classes,
  by section of the specification, most with a module of the same name for
  their free functions
- `interning.py`: the table of nodes, their canonical text and the seed-free
  digest
- `substitution.py`: traversal, rebuilding, freshening and simultaneous
  substitution
- `passes/commuting.py`: the commuting node classes, into which an evaluated
  node moves
- `passes/derivative_table.py`: the derivatives of the node classes, shared by
  passes 2 and 5

`uflx_oracle` evaluates an expression from the definitions of its nodes, with
finite differences for derivatives, quadrature for integrals and numeric maps
for pull-backs, so it does not use the passes. The value of an
`AbstractExtent` is bound in the environment, as a solver gives it to a kernel. The oracle
evaluates the classes of `uflx_toy` too, and reads the values of an
`IndexTable`, which the specification leaves to a library.

The tests in `test/` check on random expressions that a pass keeps the value
and the static type, that its output is in its grammar, and that the grammar
rejects its inputs. The construction rules of the argument spaces are checked in the
tests, by `ill_typed` in `test/testutil.py`, on the input and the output of
every pass, since the constructors do not validate. `test_controls.py` breaks the passes on purpose and checks
that the tests notice. Each random test also runs as a script, with the number
of trials as its argument, and prints its counts.

```
pip install -e ".[test,lint]"
pytest                # on 6 workers, set in pyproject.toml
ruff check . && ruff format --check .
coverage run -m pytest && coverage report
python test/test_full.py 100
```

## Differences from the specification

The classes, their fields, the passes and the helpers use the names of the
specification. The core still simplifies the specification in these places,
which the toy replaces with the classes of the specification one at a time:

- The constructors do not validate their arguments (rule 8).

The signature test reads the sibling `uflx/design/LANGUAGE.md` by default.
For an isolated checkout, set `UFLX_SPEC_PATH` to the specification path to run
that check instead of skipping it.
