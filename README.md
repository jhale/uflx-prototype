# UFLx spec interpreter

`lower(f)` runs the standard core recipe and checks its core-normal-form output.
Individual transformations remain available. Internal checks in `_src/contracts.py`
return offending nodes and diagnostic reasons without running transformations or
checking later compiler capabilities.

`gateaux_derivative(...)` constructs a `Form` from `DirectionalDerivative`
expressions. There is no separate Gateaux node or pass.
Form linearity is a declared mathematical precondition; the interpreter does not
include a linearity analyser.

`image(x, Q)` constructs a `Mesh` from topology-supported coordinates with no
free indices. Mesh integrals retain free integrand indices and sum over internal
cell indices. Cell-dependent data are fields on the topology or mesh; use mapped
reference-domain integrals and explicit index sums for other integration families.


A toy implementation of the evaluation passes in the UFLx language specification
(`design/language.md` in the uflx repository), tested against an independent oracle.

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

The optional C backend, `uflx_compiler.c_code.c_kernel`, emits a scalar kernel
for one cell. Index sums become nested C loops; input-independent subgraphs
become constant multidimensional tables. Temporary expressions are shared only
within scopes where their indices are defined. The runtime arrays keep the flat
pointer interface: `read(label, offset)` must accept an integer or a C expression
for the offset, for example `f"w[{offset}]"`. The backend retains its real-valued,
static-extent arithmetic subset and accumulates into `A[0]`.

The standard recipe in `_src/passes/lowering.py` coordinates:

| Transformation | Module |
| --- | --- |
| Apply actions | `passes/actions.py` |
| Expand derivatives and finish resulting actions | `passes/derivatives.py` |
| Expand mesh integrals | `passes/integrals.py` |
| Alternate gradients and pull-backs until stable | `passes/lowering.py`, using `gradients.py` and `pullbacks.py` |
| Check core normal form | `contracts.py` |

The shape extension uses its own recipe: expose geometry, apply shape derivatives,
then re-enter checked core lowering. Quadrature and tabulation follow that boundary
in the compiler and impose their own requirements.

The core splits a public API from a private one, as JAX does with `jax._src`.

The public API is `uflx_spec_interpreter/__init__.py`, which re-exports the
free functions of the specification and declares them in `__all__`, with the
literal classes, `RD`, scalar types and the convenience helper `num`. A user,
and a test that builds expressions, imports only these, as in
`from uflx_spec_interpreter import add, index`. `AbstractExpression` also has
the operator overloads of the specification, which call these free functions
and reject Python numbers.

Everything else is in the private package `_src`, which only the toy
extension, the oracle and the white-box tests import. The toy imports only
`_src.nodes`, for the base classes it extends. The free functions are in the
modules of `_src`:

- `literals.py`: `num`, which builds the literal of an `int` or a `float`, and a
  complex `Complex` literal, and `zero`
- `arithmetic.py`: `add`, `subtract`, `negate`, `multiply`, `divide`, `power`,
  the elementary functions and `total`, where `add`, `multiply`, `divide`,
  `real_part` and `imaginary_part` read the ranks and build the compositions of
  tensor arithmetic as the specification says
- `indices.py`: `index` and `gathered_index`
- `tensors.py`: `tensor`, `indexed`, `component`, `component_tensor`,
  `index_sum`, `inverse`, and the matrix compositions such as `transpose`,
  `matrix_product` and `determinant`
- `calculus.py`: `coordinate`, `gradient`, `directional_derivative` and
  `integral`
- `maps.py`: `domain_map` and `pull_back`
- `geometry.py`: `jacobian`, `inverse_jacobian`, `density` and the metric
- `topology.py`: `cell_index`
- `meshes.py`: `image`, `cell_map`, `side_domain`,
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
  `nodes/maps.py`, `nodes/meshes.py`,
  `nodes/topology.py`, `nodes/spaces.py` and `nodes/forms.py`: the node classes,
  by section of the specification, most with a module of the same name for
  their free functions
- `interning.py`: the table of nodes, their canonical text and the seed-free
  digest
- `traversal.py`: graph traversal and reconstruction
- `substitution.py`: capture-avoiding freshening and simultaneous substitution
- `rewriting.py`: replacement compatibility, index completion and promotion
- `contracts.py`: evaluation-stage requirements, including core normal form
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
every pass, in addition to construction validation. `test_controls.py` breaks the passes on purpose and checks
that the tests notice. Each random test also runs as a script, with the number
of trials as its argument, and prints its counts.

```
pip install -e ".[test,lint]"
pytest                # on 6 workers, set in pyproject.toml
ruff check . && ruff format --check .
pyright
coverage run -m pytest && coverage report
python test/test_full.py 100
```

## Following the specification

The implementation follows `uflx/design/language.md` and its linked chapters,
in their listed order. The signature test reads those chapters, checks that all
specified free functions are exported, and verifies their parameter names. Set
`UFLX_SPEC_PATH` to the specification's entry page for an isolated checkout.

The static type uses `domain`. A pull-back stores its `map`, an integral stores
its `integration_domain`, and a push-forward stores its `mesh`. Topology and mesh
cells both use `CellMap`, retaining the selected cell through substitution.
`Complex` and `Determinant` are nodes; tensor indexing uses ordinary index tuples,
including the specified empty-tuple factory identities.

Node classes check their own construction rules, using shared structural helpers
in `_src/validation.py`. These checks do not prove mathematical hypotheses
or form linearity. Downstream libraries remain responsible for their own leaf
contracts. `ScalarType.real` and `ScalarType.complex` are plain string constants, so they
satisfy the construction-argument whitelist. The toy's factories encode external enum values
and optional domains using permitted graph arguments.

`replace` traverses construction dependencies and preserves lexical binding;
evaluation traverses only value operands. Rewrite completion preserves complex
arithmetic and family indices. The oracle and compiler handle complex literals,
determinants and explicit cell reads independently of the rewrite rules.

Test utilities supply the default transformation laws and side selections used
by older examples. The public core API takes these arguments explicitly.
`num` is likewise an explicit interpreter convenience: core factories and
operator overloads never convert Python numbers.

This remains a small interpreter, not a production implementation. Some derived
properties are computed lazily and cached rather than eagerly during construction;
the existing pass functions implement their rule tables directly rather than
exposing a general extension-table dispatcher. Library topology contracts and
numerical capabilities are tested through the supplied toy implementations.

Each node declares `value_fields` for evaluation and `index_fields` for value
instantiation. Construction dependencies include every annotated argument, including
spaces and identity labels. Library expression terminals explicitly declare
`is_terminal = True`; an identifier namespace does not make an operation terminal.
Unknown extension operations require an extension recipe before core lowering.

Directional differentiation evaluates undifferentiated operands at the point and
inserts directions unchanged. `apply_derivatives` does not collect scalar algebra.
The compiler explicitly requests the optional `uflx_compiler/algebra.py` normalizer.
Both that optimizer and compiler simplification complete each rewritten child's
static type before rebuilding its parent. Compiler index instantiation uses the
core's capture-avoiding value substitution, followed by folding known toy tables.

## Static type checking

Install the `lint` extra and run `pyright` from the repository root. The strict
configuration includes only `src/uflx_spec_interpreter`; the compiler, oracle,
toy library and tests are excluded. No diagnostic categories are disabled.

Python argument types are now checked statically instead of repeatedly checked
inside constructors and passes. Construction still enforces the primitive argument
whitelist, finite floats, shapes, extents, domain compatibility and binder rules.
The excluded packages must continue to supply values satisfying the annotated
contracts. Wrong classes from untyped callers no longer consistently produce the
specification's mandated `TypeError`; this tradeoff is recorded in the design
repository's `possible-issues-with-spec.md`. The negative typing fixture in `test/typecheck/arguments.py` verifies
wrong argument types, rewrite callback results and non-exhaustive matches; pytest
runs it separately when Pyright is installed.

Annotations use concrete classes and tuples. The only new alias is the recursive
`Construction` type. Casts are concentrated at the reflection, cache and graph
reconstruction boundaries. Ten stored field declarations have narrow
`reportIncompatibleMethodOverride` suppressions because `Base` generates their
read-only properties at runtime. Unnecessary suppressions are reported as errors.
Cached attributes use ordinary properties with a small per-node cache decorator,
so their overrides are checked normally.

Pyright checks exhaustiveness for finite unions. `AbstractExpression` remains
open to library extensions, so core passes explicitly retain or reject unsupported
nodes; static checking cannot prove coverage of every possible extension class.

### Arithmetic construction convention

Use `add`, `multiply`, `divide`, `real_part` and `imaginary_part` (or their
operator overloads) when constructing arithmetic. They handle tensor operands
and reduce to scalar nodes for scalar operands. Direct construction of `Sum`,
`Product`, `Division`, `RealPart` and `ImaginaryPart` is checked by
`test/test_construction_boundaries.py`. Node classes remain available for
pattern matching, type annotations and inspection.

The only direct-construction exceptions are the `add`, `multiply` and `divide`
implementations in `_src/arithmetic.py`: they build the scalar nodes that
implement the lifting itself. Exceptions are specific to a function and node
class, and the check rejects unused exceptions. Other node classes are not
banned; this convention targets operations with tensor-aware arithmetic
wrappers.

The AST check follows direct imports, relative imports and import aliases,
including module aliases. It is a source convention check, not a Python
interpreter or a shape proof: wildcard imports, dynamically retrieved
constructors and constructor reassignment are outside its supported syntax.
Run it with `pytest test/test_construction_boundaries.py`; it also runs with
the ordinary package test suite.

### Source architecture checks

The ordinary test suite also checks three source conventions:

- `test/test_boundaries.py` resolves absolute and relative imports, including
  `from package import submodule`, local imports and imports under
  `TYPE_CHECKING`. It enforces the existing package dependency direction and
  prevents core implementation modules from importing the public facade,
  `_draft`, or `_src/extensions`. Extension modules may depend on the core. It also keeps static queries,
  traversal, rewrite completion and substitution independent of passes and
  contracts, and prevents transformations from importing their lowering recipe.
- `test/test_overloads.py` checks that expression operator methods return calls
  to their designated free functions with the correct operand order. Indexing
  preserves the `_members(entries)` conversion. Returning `NotImplemented` for
  unsupported operand types remains allowed. Numerical overload tests remain
  responsible for behaviour.
- `test/test_signatures.py` checks explicit facade imports against the literal
  `__all__` list, rejecting duplicate bindings, duplicate exports, wildcard
  imports and missing names. With the language specification available, its
  generated free-function and pass tables determine the exact public name set,
  plus the explicitly listed public constructors and conveniences. Existing
  signature checks also verify parameter names and supported overloads.

These checks inspect static source syntax; they do not follow dynamic imports
via `importlib` or prove runtime behaviour. The specification comparison keeps
its existing skip when the sibling specification is unavailable; set
`UFLX_SPEC_PATH` to supply it from another location. The other checks run
without the specification.

### Public and private names

The package `__init__.py` defines the public free functions and convenience
constructors. `_src/nodes/__init__.py` exposes the spec node classes used by
extensions and passes. Those names retain the spelling in the specification.

Other module-level functions, implementation classes, constants and typing
helpers begin with `_`, including helpers shared between implementation
modules. Internal `__all__` lists explicitly permit that sharing without
weakening strict Pyright's private-access checks; they do not expand the
package's public API. Private convenience helpers live in their implementation modules.

Node fields and protocol methods retain their existing names because they
form the extension/reconstruction interface. Nested functions and local
variables are already confined to their enclosing function. Imported public
classes and free functions keep their names; we do not add aliases merely
to hide imports. `test/test_boundaries.py` checks the naming convention for
module-level definitions and bindings under `_src`.
