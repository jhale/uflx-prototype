# UFLx spec interpreter

The `uflx-spec-interpreter` distribution provides three Python packages. Import
language operations from the `uflx` submodules, use `uflx_basix` for concrete
reference cells, finite elements, interpolation and quadrature, and evaluate expressions with `uflx_oracle`.
The experimental form compiler and its tests live on the `jhale/experimental-compiler` branch.

| Public module | What to import |
| --- | --- |
| `uflx.arithmetic` | Numeric literals, arithmetic and comparisons, including `num`, `add` and `less_than` |
| `uflx.tensors` | Indices, tensors and contractions, including `index`, `tensor` and `index_sum` |
| `uflx.forms` | Functions, spaces and form binding, including `function`, `form`, `partial` and `apply` |
| `uflx.calculus` | Coordinates, derivatives and integrals, including `gradient` and `integral` |
| `uflx.geometry` | Meshes, maps and geometric transformations, including `mesh` and `pull_back` |
| `uflx.passes` | Lowering and individual evaluation passes, including `lower` and `apply_derivatives` |
| `uflx.nodes` | Node classes used in pattern matching and extensions, including `Sum` and `AbstractExpression` |
| `uflx.algorithms` | Graph traversal, comparison and rewriting, including `bottom_up` and `replace` |

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

Call the optimiser explicitly on the expression to simplify.

```python
from uflx._src.optimisation import OptimisationOptions, optimise

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

The initial supported fragment is exact integer/rational scalar algebra, typed zeros,
single-index sums with static extents, vector indexing and guarded identity-tensor
contractions. Egglog also represents conjugation, real and imaginary parts,
powers, absolute values and logarithms. Rules include double conjugation,
real/imaginary projections, `x**1`, guarded `x**0`, `abs(abs(x))` and `log(1)`.
Conjugation may wrap opaque tensor or form values.

Structural cleanup selects known tensor entries, scalar broadcasts and zero
components. Component-tensor selection uses simultaneous, capture-avoiding index
substitution. It supports known-defined bodies and direct indexing permutations
that retain the underlying tensor; other potentially singular bodies are left
unchanged. Gathered indices and substitutions into stored metadata remain outside
this initial scope.

Integer and rational coefficients share an arbitrary-precision exact representation.
For example, `(2*x)/2` can become `x`, and `x/2 + x/3` can become `(5/6)*x`.
Division by an unknown or zero denominator is retained; fractions are never
converted to approximate floating-point constants. These arithmetic rewrites
are disabled by `preserve_floating_point_evaluation=True`.

The optimiser also folds determinants and nonsingular inverses of stored square
matrices up to 3 by 3, using exact integer/rational arithmetic. It reads entries
through tensor construction, indexing and broadcasts when the source is known
to be defined. Empty matrices and family indices are preserved. Singular inverses,
floating-point entries, nonzero complex entries, rectangular and larger matrices
remain untouched. This constant folder is outside the scalar e-graph and is
disabled by `preserve_floating_point_evaluation=True`. Its results feed back into
egglog, so `log(det(I))` becomes zero, including for repeated matrix families.

Other operations remain opaque in egglog; UFLx traversal can still optimise their
value operands. General tensor arithmetic should be core-lowered first.
Complex constants remain opaque.

Rules that discard an operand, such as multiplication by zero, require evidence
that the operand is mathematically defined. Terminals denote supplied, defined
values on admissible inputs; this includes runtime arrays and functions. Facts propagate through sums and products, including
terms created during search. Nonterminal extension operations remain unknown; no
user-assumptions API is implemented. Thus `x-x` and `0*x` can become zero for a
runtime input, while `0*(1/x)` stays unchanged unless the denominator is known
nonzero. Defined inputs do not make logarithms or matrix inverses total.
Delta contraction likewise requires the remaining summand to be known to be defined. With the
`indexed` group enabled, literal identity entries contract against vector or
matrix entries and compound products/summands. Capture-avoiding UFLx substitution
handles nested and multi-index sums outside the scalar e-graph; the result feeds
back into egglog. These contractions are disabled when floating-point evaluation
is preserved.

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
inside opaque nodes. This mode runs structural cleanup and safe unary egglog
rules, keeping arithmetic, powers, logarithms and literals opaque. It skips
arithmetic finishing cleanup. Unary searches appear in the region reports. It makes no guarantees
about later compiler passes or generated code.

Iteration and input-region limits are enforced; neither is a time or
e-graph-memory limit. Each region reports its exploratory search stop
(`saturated`, `iteration_limit` or `region_limit`), timings and alpha merges. A
non-expanding constant cleanup phase follows search when floating-point changes
are allowed, so the search status does not claim global saturation after cleanup.
Oversized regions fall back to optimising their children, sharing traversal for
repeated subexpressions. If those changes bring the parent within the limit, it
is searched again. The limit applies to each search, not the total work across
subregions. Oversized-region counts stop at the limit plus one. Traversal remains
recursive in this initial implementation.

The loop cost is a tree arithmetic estimate weighted by static sum extents;
opaque operations have zero estimated internal cost. It does not model DAG reuse,
memory traffic or vectorisation. Alpha sharing therefore does not guarantee that
extraction chooses `2*a` over `a+a`.

Optimisation tests skip when egglog is absent. Install both `lint` and
`optimisation` on Python 3.12+ to type-check the optional subsystem. Importing
the core or the optimisation entry point does not import egglog.

`directional_derivative(f, x, h)` takes the derivative with respect to `x` at `x`
in direction `h`. Use `directional_derivative(f, x, a, h)` to evaluate it at `a`.
Both construct the full real differential, including complex and mixed variables.
Bind directions with `form`; complex-linear and antilinear projections are
explicit compositions. The former `gateaux_derivative` helper is removed.
Form linearity is a declared mathematical precondition; the interpreter does not
include a linearity analyser.

`mesh(x, Q)` constructs a `Mesh` from topology-supported coordinates with no
free indices. Mesh integrals retain free integrand indices and add one contribution per fixed local cell slot. Cell-dependent data are fields on the topology or mesh; use mapped
reference-domain integrals and explicit index sums for other integration families.


The core implements the evaluation passes in the UFLx language specification
(`design/language.md` in the uflx repository). The tests check the passes against
an independent numerical reference interpreter.

The oracle imports the Basix extension and the core language. The Basix
extension imports the core. `test/uflx/test_boundaries.py` checks this order.

- `uflx` implements the core language and its transformations.
- `uflx_basix` attaches Basix reference cells, finite elements and basis values
  to UFLx expressions. Its two operations use reference-cell data.
  `interpolate(f, V)` uses the element’s sampling points and interpolation matrix.
  `apply_quadrature` turns one reference integral into a weighted sum.
- `uflx_oracle` evaluates the resulting expressions numerically.

### Interpolation on a reference cell

```python
import basix

from uflx.calculus import coordinate
from uflx_basix.cells import cell
from uflx_basix.spaces import space
from uflx_basix.interpolation import interpolate

K = cell(basix.CellType.triangle)
V = space("P1", K, basix.ElementFamily.P, 1)
x = coordinate(K)
p = interpolate(x[0] * x[0], V)
```

Here `p` is the linear field on the reference triangle that agrees with
`x[0] * x[0]` at its three vertices. For a physical field, the caller selects
the cell and maps the field to reference coordinates, including any Piola
transformation required by the element. Interpolation supports value sampling.

### Choosing quadrature per integral

Use `basix.make_quadrature` to choose the rule. Pass its points and weights
to `apply_quadrature` to express the integral as a weighted sum in UFLx.
Here `mass` is an integral on the reference triangle `K`.

```python
import basix

from uflx_basix.quadrature import apply_quadrature

points, weights = basix.make_quadrature(K.cell_type, degree=2)
sampled_mass = apply_quadrature(mass, points, weights)
```

To sample particular integrals in a larger expression, use their nodes as keys.
Here `load` stays available for a later choice of rule.

```python
from uflx.algorithms import substitute_values

stiffness_rule = basix.make_quadrature(K.cell_type, degree=4)
partial = substitute_values(
    mass + stiffness + load,
    {mass: sampled_mass, stiffness: apply_quadrature(stiffness, *stiffness_rule)},
)
```

For a common degree across a form, use the core traversal. It visits nested
integrals from the inside out. Point integrals are already exact.

```python
from uflx.algorithms import bottom_up
from uflx.nodes import Integral
from uflx_basix.nodes import BasixCell


def sample(node):
    if isinstance(node, Integral):
        K = node.integration_domain
        if isinstance(K, BasixCell) and K.dimension > 0:
            points, weights = basix.make_quadrature(K.cell_type, degree=2)
            return apply_quadrature(node, points, weights)
    return node


sampled_form = bottom_up(reference_form, sample)
```

The oracle can evaluate the weighted sum directly. Use
`uflx.passes.expand_pull_backs` to expand its point evaluations.

### Local coefficients and library assembly

`BasixSpace.cell_dofs` describes a packed local layout, and `num_dofs` is its
size. Two P1 triangle slots have rows `((0, 1, 2), (3, 4, 5))`, even if the
topology identifies their vertices. The Basix `discontinuous` option selects
the element. A library DOF map specifies sharing across cells.

The assembler supplies one local coefficient per entry in that layout. For
two P1 triangles with vertex rows `(0, 1, 2)` and `(0, 2, 3)`, it can gather
scalar nodal values with `assembled_values[[0, 1, 2, 0, 2, 3]]`. The assembler
also handles orientation and the accumulation of local contributions.

Geometry data follow the same packing contract. For a quadratic triangle mesh
in two dimensions, each cell supplies six pairs of coordinates in the coordinate
element's DOF order. The oracle reads them through the ordinary coefficient binding.

```python
X = space("geometry", topology, basix.ElementFamily.P, 2, shape=(2,))
x = function(X, "x")
domain = mesh(x, EuclideanSpace(2))
area = Interpreter({x: packed_coordinate_dofs}).evaluate(integral(num(1), domain))
```

The library supplies compatible coordinates and facet correspondence across
cells with matching traces. The library supplies outward conormals separately
from side names. Each selected mesh cell determines the reference cell for its
local expression.

The standard recipe in `_src/passes/lowering.py` coordinates:

| Transformation | Module |
| --- | --- |
| Expand applications | `passes/applications.py` |
| Expand derivatives and resulting applications | `passes/derivatives.py` |
| Expand mesh integrals | `passes/integrals.py` |
| Alternate gradients and pull-backs until stable | `passes/lowering.py`, using `gradients.py` and `pullbacks.py` |
| Check core normal form | `passes/contracts.py` |

Standard derivative evaluation exposes dependent geometry before differentiation
and rejects unresolved dependencies. Inner derivatives are resolved first.
Quadrature follows the checked core boundary. Geometry derivatives use the core
`directional_derivative` and `lower` operations.

### Full application and named binding

Full application constructs a lazy node. Partial application and rebinding transform explicit binders.

```python
from uflx.forms import apply, partial, rebind

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
`expand_applications` expands each `Apply` using simultaneous substitution.

`Boolean`, `less_than` and `conditional` support symbolic selection.
Eager evaluation selects a branch for each free-index assignment. A traced
predicate produces a JAX conditional, with both branches traced.
Library operations can use `_src/extensions/external_operators.py` to declare an
operand signature and a derivative action. Its `apply_external_derivatives` recipe
passes the library all point values followed by all direction values in operand order.

Core binding algorithms use the same declarations, scopes and ordered signatures
for forms, index binders and derivative variables. Capture avoidance and explicit
renaming share scope reconstruction. `alpha_equivalent` compares lexical addresses
with context-sensitive memoization, preserving free identities and structural
parameter types without changing interning or structural equality. Derivative
points and directions remain outside the variable scope. `test_shared_binding.py`
checks these laws across all four core binder kinds and mixed nesting.

`uflx.algorithms.alpha_equivalent` compares expressions modulo bound names.
The optimiser also uses an internal fingerprint to filter candidate matches.

```python
from uflx._src.algorithms.alpha import AlphaHasher, alpha_hash
from uflx.algorithms import alpha_equivalent

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

The `test_*_exploration.py` files exercise these boundaries. An affine-interval
Hermite reconstruction probe checks physical endpoint slopes and differentiation
of the geometry-dependent coefficient transformation. It is not a general Hermite element implementation.

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
and direction functions to obtain real matrices. The oracle tests the mixed
differential and its explicit parts against finite differences.

`uflx.calculus.reference_integral(f, D, cell_slot)` gives one local cell’s reference integral.
Mesh expansion adds these integrals over the fixed slots without introducing
an array index or cell sum.
It works for cells and facet meshes. `CellMap(parent, cell_slot)` selects a slot;
`SideMap(parent, source, entries)` stores one `(parent_slot, inclusion)` per
source slot. The factory accepts `side_map(E, (cell, A))`, inferring the parent from
the `CellMap` handle `cell`. All entries must share that parent. Topologies carry no
side or normal data. `conormal(relation, m0, ...)` supplies one oriented reference
conormal per source slot, independently of map identity. Successive restrictions
use nested pull-backs: `pull_back(pull_back(u, outer), inner)`.
The optional `side_domain(D, S, *entries)` helper derives source geometry; a
source mesh can also be constructed independently. Cell bindings and neighbour
lookup belong to the calling finite element package.
`AbstractExpression` also has
the operator overloads of the specification, which call these free functions
and reject Python numbers.

### Writing a transformation pass

Import node classes from `uflx.nodes` for pattern matching, annotations and
extension base classes. Import graph operations from `uflx.algorithms`. For
example, this pass folds integer sums after visiting their operands.

```python
from uflx.algorithms import bottom_up
from uflx.arithmetic import add, num
from uflx.nodes import Integer, Sum


def fold_integer_sum(node):
    match node:
        case Sum(Integer(a), Integer(b)):
            return num(a + b)
    return node


assert bottom_up(add(num(2), num(3)), fold_integer_sum) == num(5)
```

For a recursive pass, `memoized` shares results across visits, `map_values`
rebuilds a node with its transformed value operands, and `complete_indices`
restores free indices that a replacement dropped. Use `uflx.algorithms.replace` to
substitute through construction dependencies or `uflx.algorithms.substitute_values`
to instantiate values while keeping geometry and identity metadata fixed.

### Core implementation

The private `_src` modules implement the public functions.

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
- `geometry.py`: `jacobian`, `density` and the metric
- `meshes.py`: `mesh`, `cell_map`, `side_domain`,
  `side_map` and `conormal`
- `spaces.py`: `conjugate_space` and `dual_space`
- `forms.py`: `function`, `form` and `apply`
- `algorithms/form_binding.py`: `partial` and `rebind`
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

### Numerical reference interpreter

`uflx_oracle.interpreter.Interpreter` evaluates the original expression tree.
Use it to obtain a value for a field, apply a form to numerical arguments, or
compare an expression before and after lowering. Install `pip install -e ".[oracle]"` for
the JAX evaluator and Basix. For an integral on a positive-dimensional Basix cell,
the oracle calls `basix.make_quadrature(K.cell_type, degree=12)`. Other reference
cells can supply their own `quadrature(degree)` method. A point integral uses one
sample of weight one. The oracle uses JAX's active precision. Set
`JAX_ENABLE_X64=true` for double-precision reference values.

```python
from uflx.arithmetic import num
from uflx.calculus import coordinate, gradient
from uflx.nodes import EuclideanSpace
from uflx_oracle.interpreter import Interpreter

plane = EuclideanSpace(2)
x = coordinate(plane)
f = x[0] ** num(2) + x[1]
interpreter = Interpreter()
point = {plane: [3.0, 4.0]}
interpreter.evaluate(f, points=point)  # 13.0
interpreter.evaluate(gradient(f, plane), points=point)  # [6.0, 1.0]
```

Pass field data to `Interpreter(environment)`, keyed by function nodes. A field can
be a numerical value, a UFLx expression, or a callable of the current coordinate.
The oracle can read the same field more than once during evaluation, so callbacks
should be pure. Basix fields accept local coefficient arrays. A domain-free library
terminal with no indices or form slots can be bound to an array. Such terminals
and runtime extents accept node keys or their library-defined labels.
Use `arguments=(u_value, v_value)` for ordered form slots. A tuple-valued slot
takes a tuple. Free indices appear as leading axes of the result, in the order
of `expression.free_indices`, followed by the tensor value shape. Pass
`indices={i: k}` to select entry `k` on axis `i`.

The assembler selects the physical cell or facet and supplies its local data.
A field bound to a callable takes just the evaluation point. When the expression
uses a cell map or side map, the map identifies the local slot to read. A mesh
integral visits each slot in its declared local configuration.

For direct point evaluation, pass `cells={domain: slot}` when a coefficient
or chart must be selected from several local slots. A single-cell domain uses
slot zero. On an interior facet, each side map selects its own parent slot so
both traces can be evaluated at the same point. An external operator that receives a
whole field can sample it as `field(point)` on the selected slot.

The oracle walks the original UFLx graph using JAX operations. For example,
`index_sum(A[i, j] * b[j], (j,))` evaluates to the array given by
`jnp.einsum("ij,j->i", A_values, b_values)`. A spatial derivative differentiates
a function of the point. For a Basix field, the oracle contracts the table of
exact basis derivatives with the reference direction. A directional derivative
of a UFLx expression varies the named functions simultaneously.

For a family of indexed pullbacks, the oracle can use `jax.vmap` to sample the
field across its free indices. Eager conditionals select their branch separately
at each sample. JAX can also differentiate an entire evaluation with respect to
its inputs.

```python
import jax
import jax.numpy as jnp


def at(p):
    return interpreter.evaluate(f, points={plane: p})


value, derivative = jax.jit(jax.value_and_grad(at))(jnp.array([3.0, 4.0]))
# value = 13, derivative = [6, 1]
```

Bind an external operation to a JAX-traceable function. A field operand is a
callable of its coordinates, so a binding can sample the same field at several
points. Use `Pointwise` when the function takes operand values at the current
point. JAX differentiates through either calling convention.

```python
import jax.numpy as jnp
from uflx._src.extensions.external_operators import external_operator
from uflx_oracle.interpreter import Interpreter, Pointwise

response = external_operator(law, (x[0],))
interpreter = Interpreter({law: Pointwise(jnp.sin)})
interpreter.evaluate(gradient(response, plane), points=point)  # [cos(3), 0]
```

If a library needs a custom derivative, it can define one on the JAX function
with `jax.custom_jvp`. If you explicitly expand a UFLx derivative with
`apply_external_derivatives`, bind the derivative-action operation that appears
in the expanded expression.

Under JAX transformations, bindings must be pure and captured fields containing
traced inputs must be used within the trace. When tracing a conditional, supply
the bindings needed by both branches. Cell selections, element types and extents
that determine array shapes are static.

### Running the tests

Run `pytest test` for the full suite or select a package, for example
`pytest test/uflx_oracle`. Each test belongs to the lowest layer that can run its assertions and fixtures.

| Test folder | What it can use |
| --- | --- |
| `test/uflx/` | Core language and passes, with fixture spaces, runtime data and topologies |
| `test/uflx_basix/` | Core and Basix, including reference elements, interpolation and quadrature |
| `test/uflx_oracle/` | Both lower layers and oracle evaluation |

Shared fixtures follow the same import order.

The oracle tests check on random expressions that a pass keeps the value
and the static type, that its output is in its grammar, and that the grammar
rejects its inputs. The construction rules of the argument spaces are checked in the
tests, by `ill_typed` in `test/testutil.py`, on the input and the output of
every pass, in addition to construction validation. `test_controls.py` breaks the passes on purpose and checks
that the tests notice. Run a randomized test as a Python module from the
repository root to set the trial count and print its results.

```
pip install -e ".[test,lint]"
pytest                # on 6 workers, set in pyproject.toml
ruff check . && ruff format --check .
pyright
coverage run -m pytest && coverage report
python -m test.uflx_oracle.test_full 100
```

## Following the specification

The implementation follows `uflx/design/language.md` and its linked chapters,
in their listed order. The signature test reads those chapters, checks that all
specified free functions are exported, and verifies their parameter names. It also
checks concrete node names and ordered construction arguments against the class tables. Set
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
satisfy the construction-argument whitelist. The factories in `uflx_basix` encode external enum values
using permitted graph arguments.

`replace` traverses construction dependencies and preserves lexical binding;
evaluation traverses only value operands. Rewrite completion preserves complex
arithmetic and family indices. The oracle handles complex literals,
determinants and explicit cell reads independently of the rewrite rules.

Test utilities supply the default transformation laws and side selections used
by older examples. The public core API takes these arguments explicitly.
`num` is likewise an explicit interpreter convenience: core factories and
operator overloads never convert Python numbers.

This remains a small interpreter, not a production implementation. Some derived
properties are computed lazily and cached rather than eagerly during construction;
the existing pass functions implement their rule tables directly rather than
exposing a general extension-table dispatcher. Generic spaces, runtime data and topologies are test fixtures in `test/uflx/`.
The Basix tests exercise the finite-element and reference-cell adapters.

Each node declares `value_fields` for evaluation and `index_fields` for value
instantiation. Construction dependencies include every annotated argument, including
spaces and identity labels. Library expression terminals explicitly declare
`is_terminal = True`; an identifier namespace does not make an operation terminal.
Unknown extension operations require an extension recipe before core lowering.

Directional differentiation evaluates undifferentiated operands at the point and
inserts directions unchanged. `apply_derivatives` does not collect scalar algebra.

## Static type checking

Install the `lint` extra and run `pyright` from the repository root. The strict
configuration covers `src/uflx`. No diagnostic categories are disabled.

Python argument types are now checked statically instead of repeatedly checked
inside constructors and passes. Construction still enforces the primitive argument
whitelist, finite floats, shapes, extents, domain compatibility and binder rules.
The excluded packages must continue to supply values satisfying the annotated
contracts. The negative typing fixture in `test/uflx/typecheck/arguments.py` verifies
wrong argument types, rewrite callback results and non-exhaustive matches; pytest
runs it separately when Pyright is installed.

Annotations use concrete classes and tuples. `ExpressionScalarType` includes
Boolean, real and complex values. Function spaces and numerical operations use
`NumericalScalarType`, which admits real and complex values. Casts are concentrated at the reflection, cache and graph
reconstruction boundaries. Stored properties return typed dataclass fields without
casts or Pyright suppressions. Computed attributes use ordinary properties with a
small per-node cache decorator. `test/uflx/test_node_declarations.py` checks metadata
names and the correspondence between private fields and public getters.

Pyright checks exhaustiveness for finite unions. `AbstractExpression` remains
open to library extensions, so core passes explicitly retain or reject unsupported
nodes; static checking cannot prove coverage of every possible extension class.

### Arithmetic construction convention

Use `add`, `multiply`, `divide`, `real_part` and `imaginary_part` (or their
operator overloads) when constructing arithmetic. They handle tensor operands
and reduce to scalar nodes for scalar operands. Direct construction of `Sum`,
`Product`, `Division`, `RealPart` and `ImaginaryPart` is checked by
`test/uflx/test_construction_boundaries.py`. Node classes remain available for
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
Run it with `pytest test/uflx/test_construction_boundaries.py`; it also runs with
the ordinary package test suite.

### Source architecture checks

The ordinary test suite also checks three source conventions:

- `test/uflx/test_boundaries.py` resolves absolute and relative imports, including
  `from package import submodule`, local imports and imports under
  `TYPE_CHECKING`. It enforces the existing package dependency direction and
  prevents core implementation modules from importing the public submodules,
  `_draft`, or `_src/extensions`. Extension modules may depend on the core. It also keeps static queries,
  traversal, rewrite completion and substitution independent of passes and
  contracts, and prevents transformations from importing their lowering recipe.
- `test/uflx/test_overloads.py` checks that expression operator methods return calls
  to their designated free functions with the correct operand order. Indexing
  preserves the `_members(entries)` conversion. Returning `NotImplemented` for
  unsupported operand types remains allowed. Numerical overload tests remain
  responsible for behaviour.
- `test/uflx/test_signatures.py` checks each public submodule's imports against its literal
  `__all__` list, rejecting duplicate bindings, duplicate exports, wildcard
  imports and missing names. With the language specification available, its
  generated free-function and pass tables determine the names exported across the
  language submodules alongside their convenience functions. The
  signature checks also verify parameter names and supported overloads.

These checks inspect static source syntax; they do not follow dynamic imports
via `importlib` or prove runtime behaviour. The specification comparison keeps
its existing skip when the sibling specification is unavailable; set
`UFLX_SPEC_PATH` to supply it from another location. The other checks run
without the specification.

### Public and private names

Each public submodule declares its exports in `__all__`. Importing `uflx` alone
loads only the package root. Node classes retain the spelling in the specification.
The Basix extension uses these public imports.

Shared interfaces have ordinary names within `_src`. Module-local helpers begin
with `_`. Strict Pyright checks cross-module access.

Node fields and protocol methods retain their existing names because they
form the extension/reconstruction interface. Nested functions and local
variables are already confined to their enclosing function. Imported public
classes and free functions keep their names; we do not add aliases merely
to hide imports. `test/uflx/test_boundaries.py` checks cross-module private imports
and the dependency layers.
