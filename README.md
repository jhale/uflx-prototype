# UFLx spec interpreter

`lower(f)` runs the standard core recipe and checks its core-normal-form output.
Individual transformations remain available. The [rule map](RULES.md) tracks the
mathematical groups and their implementations. Internal checks in `_src/passes/contracts.py`
return offending nodes and diagnostic reasons without running transformations or
checking later compiler capabilities.

Tensor addition, scalar multiplication and division retain their arithmetic nodes.
`expand_arithmetic(f)` converts these to scalar components. `lower` runs it after
spatial rewriting, so compilers still receive scalar arithmetic.

The implementation in `_src/` is organised by responsibility.

| Location | Contents |
| --- | --- |
| Base directory | Language-facing factory functions |
| `nodes/` | Node definitions, construction validation, interning, static queries and argument types |
| `algorithms/` | Traversal, rewriting, substitution and generated names |
| `rules/` | Mathematical rewrite rules |
| `passes/` | Pass recipes and output contracts |
| `extensions/` | Optional language extensions |
| `optimisation/` | Optional egglog encoding, algebraic rules, extraction costs and reports |

The optional optimiser requires Python 3.12+ and the `optimisation` extra:

```sh
pip install -e ".[optimisation]"
```

It is invoked explicitly and is not yet wired into the compiler pipeline:

```python
from uflx_spec_interpreter._src.optimisation import OptimisationOptions, optimise

result = optimise(
    expression,
    OptimisationOptions(
        rule_groups=("arithmetic", "indexed"),
        preserve_floating_point_evaluation=False,  # True preserves arithmetic operations and grouping.
        iterations=3,
        max_region_nodes=60,
        cost="loops",  # Or "tree"; a custom cost_model can be passed to optimise.
    ),
)
simplified = result.expression
report = result.report
```

The initial supported fragment is exact integer/scalar algebra, typed zeros,
single-index sums with static extents, vector indexing and guarded identity-tensor
contractions. Structural cleanup also selects known tensor entries, scalar
broadcasts and zero components, removes double conjugation, and simplifies real
and imaginary parts when safe. General binder substitution is not implemented.
Other operations remain opaque in egglog; UFLx traversal can still optimise their
value operands. General tensor arithmetic should be core-lowered first.
Rational/complex constants are not yet exposed as algebraic operators. Existing
compiler cleanup remains separate; it does not inherit this optimiser's policy.

Rules that discard an operand, such as multiplication by zero, require evidence
that the operand is mathematically defined. Facts come from literals and a small
set of operations with known inputs. Runtime terminals and extension nodes are
unknown; no user-assumptions API is implemented. In particular, `0*(1/x)` stays
unchanged for unknown `x`, while `0*(1/2)` can simplify to zero. Delta contraction
likewise requires the vector's components to be known to be defined.

Opaque expressions are interned using alpha fingerprints followed by exact
alpha comparison. Their free identities remain significant, and reconstruction
retrieves an original UFLx representative. No cross-invocation cache or general
scope-normalised e-graph is implemented. With only the `arithmetic` group selected,
index sums are opaque and can share modulo renaming of their own bound indices.
`PyObject` is deliberately not used: its serialised snapshots do not preserve
UFLx's original interned object references.

These are exact real/complex algebraic transformations, not guarantees of identical
floating-point execution. Set `preserve_floating_point_evaluation=True` to retain arithmetic
operations and reduction order, including constant arithmetic and operations
inside opaque nodes. This mode runs structural cleanup only, skips both egglog
search and arithmetic finishing cleanup, and reports no search regions. It makes
no guarantees about later compiler passes or generated code.

Iteration and input-region limits are enforced; neither is a time or
e-graph-memory limit. Each region reports its exploratory search stop
(`saturated`, `iteration_limit` or `region_limit`), timings and alpha merges. A
non-expanding constant cleanup phase follows search, so the search status does not
claim global saturation after cleanup. Oversized-region counts stop at the limit
plus one. Traversal remains recursive in this initial implementation.

The loop cost is a tree arithmetic estimate weighted by static sum extents;
opaque operations have zero estimated internal cost. It does not model DAG reuse,
memory traffic or vectorisation. Alpha sharing therefore does not guarantee that
extraction chooses `2*a` over `a+a`.

Optimisation tests skip when egglog is absent. Install both `lint` and
`optimisation` on Python 3.12+ to type-check the optional subsystem. Importing
the core or the optimisation entry point does not import egglog.

`directional_derivative(f, x, a, h)` is the sole function-derivative primitive.
It gives the full real differential, including complex and mixed variables.
Bind directions with `form`; complex-linear and antilinear projections are
explicit compositions. The former `gateaux_derivative` helper is removed.
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

The C emitter, `uflx_compiler.c_code.c_kernel`, emits scalar arithmetic with
loops over `output_indices`. `index_bindings` maps remaining
indices to C expressions, such as `entity_local_index[0]`. Index sums become
nested C loops; outer integral sums surround the output loops. Input-independent
subgraphs become constant multidimensional tables. A buffered scope tree schedules
computations using their free indices and bound component positions. Structurally
equal expressions share temporaries; entire reductions and partial products move
outside loops they do not depend on. Computations reused across output
indices can be precomputed into temporary arrays before the output loops. This
also separates basis evaluation from tensor contraction. Arrays span only the
required output indices, and structurally equal expressions share them. Loads
and expressions with at most one arithmetic operation are left inline. Runtime geometry independent of quadrature
is computed once per invocation, while coefficient evaluation at each point is
shared across argument DOFs. Computations remain inside conditional branches
unless a reusable value is already available in an enclosing scope. Scalar products
are regrouped by the scope where their factors become available, so quadrature weights and geometry can be multiplied before basis
products. Small reductions (up to three entries) that select explicit tensor
components are expanded to expose shared geometry; DOF and quadrature loops are
retained. Product regrouping can change floating-point rounding. Algebraic
factorisation across sums is not performed. A single-use multiplication directly
before an accumulation is emitted in that accumulation, allowing the C compiler
to generate fused multiply-add instructions with its normal contraction settings.
Shared and hoisted products keep their temporaries. Fused operations may round
differently from separate multiplication and addition; no fast-math flags are
required. The runtime arrays keep the flat
pointer interface: `read(label, offset)` must accept an integer or a C expression
for the offset, for example `f"w[{offset}]"`. The backend retains its real-valued,
static-extent arithmetic subset. With no output indices it accumulates into `A[0]`.

`compile_ufcx(f, mesh, degree, name="my_form", coefficients=..., constants=...)`
provides the UFCx output boundary. It returns C source and the coefficient,
constant and local tensor layouts. The source includes FFCx's `ufcx.h` and exports
`my_form` (`ufcx_form`), `my_form_integral` (`ufcx_integral`) and `my_form_kernel`.
Compile it with the include directory containing your installed FFCx header.
Explicit input sequences set packing order; omitted sequences use graph traversal
order. Offsets count scalar entries in `w` and `c`.

This backend tabulates directly in standard Basix element ordering. It does not
use the toy mesh's global DOF map or tensor-product DOF reordering. A kernel
computes one cell or facet contribution, with outer loops over local argument DOFs;
the solver packs cell-local coefficients and three-component coordinate DOFs,
performs DOF transformations, and assembles the result. Mesh connectivity and
coordinate values are absent from the generated kernel. Equivalent input graphs
may still produce differently ordered arithmetic after canonicalisation.

The supported UFCx subset is float64 cell, exterior-facet and interior-facet
integrals of rank 0, 1 or 2, on one mesh with one Basix cell type. Facet kernels
currently require a uniform reference facet type. Geometry is a tensor of scalar
Basix coordinate functions (including curved geometry), or vertex coordinates interpreted as P1.
Element families and degrees follow the toy Basix spaces, with the equispaced
variant; custom variants are not yet exposed. Mixed arguments and complex inputs
are rejected. Scalar and tensor constants are packed separately from coefficients.
Blocked vector and tensor spaces use `basix_space(..., shape=(d,))` or
`shape=(d, d)` with a scalar base element. Each scalar basis DOF carries the
components consecutively, matching Basix/UFCx ordering. Arguments and
coefficients use this layout on cells and both kinds of facets; a vector form
produces one full local tensor in one kernel call. Basis and component indices
remain separate: component choices are specialised before C emission, removing
structural zeros. Component updates share a basis-pair loop and write directly
into the interleaved output. Scalar contractions collect repeated trial factors
and prepare their material/geometry-scaled coefficients outside the basis-pair
loop. Quadrature-invariant basis contractions retain their existing grouping
to preserve hoisting. In single-block kernels, small spatial contractions with
invariant transformation coefficients are collected into geometry metrics before basis evaluation;
for affine Poisson this computes the symmetric metric outside quadrature.
Varying transformations and coupled component blocks retain their existing
evaluation order. Symmetry-compressed tensor spaces and mixed elements are not
supported.

The source targets the UFCx interface in FFCx 0.12; other versions need their own compatibility validation.

For exterior facets (`ds`), construct the boundary with `side_domain` on a
one-sided topology and pull the cell integrand back through its `side_map`:

```python
boundary = side_domain(mesh, boundary_topology, "inside")
trace = pull_back(integrand, side_map(mesh, boundary, "inside"))
code = compile_ufcx(integral(trace, boundary), mesh, degree=4)
assert code.integral_type == "exterior_facet"
```

The generated kernel selects a reference facet using `entity_local_index[0]`
in Basix local-facet numbering. Its facet tables, measure and outward normals
come from reference geometry; the boundary topology's connectivity is not
embedded. Exterior facets require no quadrature permutation. Compile cell and
boundary contributions separately: one generated form currently has one
integration domain and the default integral ID `-1`.

For interior facets (`dS`), the side topology has two sides named `"+"` and `"-"`.
Use `side_domain(mesh, interface_topology, "+")` for the integration domain and
`side_map(mesh, interface, side)` to restrict each field to the chosen cell.
`code.integral_type` is then `"interior_facet"`. For example:

```python
interface = side_domain(mesh, interface_topology, "+")
plus = side_map(mesh, interface, "+")
minus = side_map(mesh, interface, "-")
jump = pull_back(u, plus) - pull_back(u, minus)
code = compile_ufcx(integral(jump * jump, interface), mesh, degree=4)
```

UFCx supplies two entries in `entity_local_index` and `quadrature_permutation`.
Permutation numbers are `2 * rotations + reflections`. The compiler precomputes
separate reference-facet and basis tables for each side, then selects them at
runtime. It does not generate kernels for every pair of facets or permutations.
Constant folding preserves output and dispatch indices to avoid materialising
cross-products of the two sides' tables.

Interior coefficients use `w[coefficient][side][local_dof]`, with `+` before `-`.
Coordinates use `coordinate_dofs[side][coordinate_dof][3]`; constants are shared
between sides. Each argument axis has twice the element dimension, again with
`+` DOFs before `-` DOFs. A bilinear kernel therefore fills all four coupling
blocks. The solver remains responsible for element DOF transformations and for
packing two conforming cells with their facet permutations.

`test/test_ufcx.py` compares generated local tensors and descriptors with FFCx,
including higher-order elements, Piola maps, curved cells and every local facet
of intervals, triangles, quadrilaterals, tetrahedra and hexahedra. Interior-facet
comparisons cover every local-facet pair and every rotation/reflection combination.
It also contains DOLFINx cell, exterior-facet and interior-facet assembly tests using the generated
`ufcx_form` address directly. These tests
skip when their optional dependencies are unavailable. With local FFCx and UFL
checkouts on `PYTHONPATH`, run `pytest test/test_ufcx.py`; `UFCX_INCLUDE_DIR` can
supply the header directory separately. Keep the compiled shared library alive
while DOLFINx uses the form's pointers.

The standard recipe in `_src/passes/lowering.py` coordinates:

| Transformation | Module |
| --- | --- |
| Apply actions | `passes/actions.py` |
| Expand derivatives and finish resulting actions | `passes/derivatives.py` |
| Expand mesh integrals | `passes/integrals.py` |
| Alternate gradients and pull-backs until stable | `passes/lowering.py`, using `gradients.py` and `pullbacks.py` |
| Check core normal form | `passes/contracts.py` |

Standard derivative evaluation exposes dependent geometry before differentiation
and rejects unresolved dependencies. Inner derivatives are resolved first.
Quadrature and tabulation follow the checked core boundary. The private shape
convenience now constructs an ordinary `DirectionalDerivative`, without an
extension node or separate lowering recipe.

### Named binding and geometry experiment

The private `_src/extensions/binding.py` module explores these operations:

```python
from uflx_spec_interpreter._src.extensions.binding import apply, partial, rebind

F = form(u * v, (u, v))
partial(F, {u: w})  # form(w * v, (v,))
partial(rebind(F, {v: p}), {u: v})  # form(v * p, (p,))
apply(F, (w, p))  # lazy full application
```

`partial` substitutes simultaneously into an explicit outer `Form` and rejects
capture of a retained argument; explicitly `rebind` that argument first. Rebinding
also supports `ComponentTensor` and `IndexSum`. Form operations preserve grouping
and conjugation and reject dependent argument spaces, including inherited slots.
Nested internal binders may freshen during substitution.
`apply` also accepts sums and nested forms with an ordered argument signature;
it uses existing `Action` nodes pending migration to the specified `Apply` node.

Core binding algorithms use the same declarations, scopes and ordered signatures
for forms, index binders and derivative variables. Capture avoidance and explicit
renaming share scope reconstruction. `alpha_equivalent` compares lexical addresses
with context-sensitive memoization, preserving free identities and structural
parameter types without changing interning or structural equality. Derivative
points and directions remain outside the variable scope. `test_shared_binding.py`
checks these laws across all four core binder kinds and mixed nesting.

The alpha utilities live in `uflx_spec_interpreter/_src/algorithms/`:

```python
from uflx_spec_interpreter._src.algorithms.alpha import AlphaHasher, alpha_hash
from uflx_spec_interpreter._src.algorithms.binding import alpha_equivalent

key = alpha_hash(expression)  # One expression, with a temporary cache.
hasher = AlphaHasher()  # Reuse summaries across a compiler pass.
if hasher.fingerprint(left) == hasher.fingerprint(right):
    same = alpha_equivalent(left, right)  # Hash collisions still require this check.
hasher.clear()  # Release cached graphs when the pass finishes.
```

The implementation uses the reference-based alpha-summary algorithm with deterministic
128-bit integer mixing, persistent occurrence maps and shared dependency sets. It
preserves opaque labels, parameter signatures, tuple grouping and conjugation.
Fingerprints are deterministic for the same construction data across processes, but
are not a versioned disk-cache format or cryptographic hashes. These are explicit
utilities; ordinary node equality, hashing and interning retain their existing meaning.
The comparison prototypes and benchmarks remain in `scratch/alpha_hash`.

Free-occurrence checks distinguish construction and value dependencies.
Declarations are not free uses, but their parameter types contribute dependencies;
labels remain opaque. Capture checks inspect construction dependencies.

Standard `apply_derivatives` selectively exposes geometry when a variable occurs
in construction metadata, then applies ordinary derivative rules. Nested points
and directions participate in outer derivatives. Unresolved geometry or space
dependence is rejected. The former private differentiation experiment delegates
to this core path.

The `test_*_exploration.py` files exercise these boundaries. Geometry checks
compare area, energy, coordinate fields, perimeter, and Raviart–Thomas mass against
finite differences, including second geometry derivatives. Mixed field/geometry
derivatives agree in both orders for energy and Raviart–Thomas mass, and with
independent finite differences. An affine-interval Hermite reconstruction probe
checks physical endpoint slopes and differentiation of the geometry-dependent
coefficient transformation. It is not a general Hermite element implementation.

The proposed form restriction is that no bound function occurs free in any exposed
argument space, including inherited slots. Spaces may depend on free geometry
parameters; differentiating those parameters still requires an explicit conversion
to fixed input spaces. A library reconstruction may use symbolic geometry-dependent
matrices rather than only a standard push-forward. Opaque geometry-dependent data
needs an explicit derivative rule or must be rejected. General element-library
integration, arbitrary derivative orders, and topology changes are not validated
by these probes. The remaining public binding API migration is separate.

### Real and complex derivative forms

`DirectionalDerivative` accepts mixed real/complex variable tuples. `Form` and
`DualSpace` require one scalar type within a product-space slot; different slots
may use different scalar types. Output scalar type does not imply complex
linearity in an argument.

The optional library composition in `extensions/complex_differentiation.py`
returns `(D[h] - i*D[i*h])/2` and `(D[h] + i*D[i*h])/2` for complex variables.
Bind a fresh direction as `h` or `conjugate(h)`, respectively. The helper neither
binds nor evaluates; it is not a core primitive. Both parts can be complex even
when the full differential is real.

Alternatively, use a real direction `q` of a compatible library space. Bind
`D[q]` and `D[i*q]` as separate real slots, adding real-variable derivative
blocks. Apply `real_part` or `imaginary_part` to bodies before binding real test
and direction functions to obtain real matrices. The mixed-block tests compare
both representations with finite differences of assembled FE residuals.

The core splits a public API from a private one, as JAX does with `jax._src`.

The public API is `uflx_spec_interpreter/__init__.py`, which re-exports the
free functions of the specification and declares them in `__all__`, with the
literal classes, `RD`, scalar types and the convenience helper `num`.
`reference_integral(f, D, group, entry)` gives one selected entity’s reference
integral. Mesh expansion wraps it in a fresh cell sum; UFCx compilation supplies
a local entry directly. It works for cells and facet meshes. A user,
and a test that builds expressions, imports only these, as in
`from uflx_spec_interpreter import add, index`. `AbstractExpression` also has
the operator overloads of the specification, which call these free functions
and reject Python numbers.

Everything else is in the private package `_src`, which only the toy
extension, compiler, oracle and white-box tests import. The toy imports only
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
- `forms.py`: `function`, `form` and `action`
- `nodes/static.py`: the free functions that read the static type

The private modules of `_src`:

- `nodes/base.py`: `Base`, which makes subclasses frozen dataclasses. Dataclasses
  supply storage, constructors and positional patterns. Private fields such as
  `_domain` implement abstract properties through typed getters. `argument_names`
  records their public names for traversal and reconstruction. The interning
  metaclass uses a deterministic digest and returns cache hits without initializing
  them again. New nodes are initialized and validated before entering the weak cache.
- `nodes/expressions.py`: `AbstractExpression`, its static-type contract, expression
  validation and Python operators.
- `nodes/structure.py`: inspection of construction dependencies and expression operands.
- `nodes/caching.py`: the shared decorator for caching computed properties.
- `nodes/literals.py`, `nodes/arithmetic.py`, `nodes/indices.py`,
  `nodes/tensors.py`, `nodes/calculus.py`, `nodes/domains.py`,
  `nodes/maps.py`, `nodes/meshes.py`,
  `nodes/topology.py`, `nodes/spaces.py` and `nodes/forms.py`: the node classes,
  by section of the specification, most with a module of the same name for
  their free functions
- `algorithms/traversal.py`: graph traversal and reconstruction
- `algorithms/substitution.py`: capture-avoiding freshening and simultaneous substitution
- `algorithms/rewriting.py`: replacement compatibility, index completion and promotion
- `passes/contracts.py`: evaluation-stage requirements, including core normal form
- `rules/`: mathematical rules grouped as in the specification; passes select and
  compose them.

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
in `_src/nodes/validation.py`. These checks do not prove mathematical hypotheses
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
`_Construction` type. Casts are concentrated at the reflection, cache and graph
reconstruction boundaries. Stored properties return typed dataclass fields without
casts or Pyright suppressions. Computed attributes use ordinary properties with a
small per-node cache decorator. `test/test_node_declarations.py` checks metadata
names and the correspondence between private fields and public getters.

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

Shared interfaces have ordinary names within `_src`. Module-local helpers begin
with `_`. The private package boundary keeps these interfaces out of the public
API; only the two facades use `__all__`. Strict Pyright checks cross-module access.

Node fields and protocol methods retain their existing names because they
form the extension/reconstruction interface. Nested functions and local
variables are already confined to their enclosing function. Imported public
classes and free functions keep their names; we do not add aliases merely
to hide imports. `test/test_boundaries.py` checks cross-module private imports
and the dependency layers.
