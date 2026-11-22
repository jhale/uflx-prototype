# Interning the nodes

> Since written, the nodes are no longer dataclasses. A class declares its
> construction arguments by annotation, `Base` interns them before
> construction, and `AbstractExpression` has no defaults, so each concrete
> class states its whole static type (see `nodes/base.py`).

A plan for interning every node of the toy, so that equal objects are the same
object, as rule 7 of the base object in LANGUAGE.md allows.

## Why

- Every node is a frozen dataclass, so `==` and `hash` walk the whole subtree.
  Pass outputs share subexpressions heavily, so each walk is exponential in
  the depth of the sharing. The surface test once ran for eight hours on this.
- Four caches are keyed on `id()` to avoid those walks: `by_identity`
  in `static.py`, `BUILT` and `VALUES` in `oracle.py`, and `contains` in
  `test/testutil.py`. With interning, a node is its own cheap key.
- Equal subexpressions built separately become one object, such as the metric
  in `density` and `inverse_jacobian`, so every pass output has maximal sharing
  and later walks visit fewer nodes.
- The cache of `_substitute` hashes its arguments, which becomes constant time.

## The rules to keep

From the rules of the base object in LANGUAGE.md:

- Rule 1: constants are compared by type and value. `1`, `1.0` and `True` are
  distinct, and `0.0` and `-0.0` are distinct. Python treats each pair as
  equal with the same hash, so the key of the intern table cannot use the
  values directly.
- Rule 4: a node is reconstructible from its class and construction
  arguments, and the reconstruction is equal to the original. With interning
  the reconstruction is the original.
- Rule 6: any order and any walk, hashing included, is deterministic across
  processes. Python's `hash` of `str` and of `bytes` is randomised per process
  by `PYTHONHASHSEED`, so neither may feed the hash of a node or the key of the
  intern table.
- Rule 7: equality is structural, and an implementation in which `is` is
  structural interns.
- Rule 10: a concrete class states an `identifier`, such as `"uflx.Sum"`, and
  the hash uses the identifier and the construction arguments, never the
  location of the class in the code.

## Design

1. Every node has a digest, computed once at construction with
   `hashlib.blake2b` (16 bytes) over a canonical encoding of the node: the
   `identifier` of the class, then the fields in order, each encoded by
   `encode`:
   - a `Node` is its own digest, already computed, so the digest of a node
     costs work proportional to its number of fields, not to its subtree;
   - a `tuple` is a tag and its length, then its items;
   - an `int`, `bool`, `str` or `None` is a tag for its type, then its value
     as UTF-8 text, with the length of a `str` so that no two encodings run
     together;
   - a `float` is a tag and `float.hex()`, which separates `0.0` from `-0.0`;
   - a `complex` is a tag and the encodings of its real and imaginary parts.
1. Every class states an `identifier`: the name of the class in the
   specification where there is one, such as `"uflx.Product"` for `Prod` and
   `"uflx.ComponentTensor"` for `CT`, and `"toy."` followed by the class name
   for a class of the toy alone, such as `"toy.CellMap"`.
1. A metaclass `Interned` on `Node`. Its `__call__` raises `TypeError` for a
   class with no `identifier`, builds the instance as now, computes the
   digest, and returns the stored instance for that digest if there is one,
   or stores and returns the new one.
1. The table is a `weakref.WeakValueDictionary` keyed by the digest as an
   `int`, whose Python hash is deterministic, so unused nodes are freed and no
   `str` or `bytes` hash is involved. A collision of 128-bit digests is
   negligible, as for content-addressed storage.
1. `__hash__` returns the digest as an `int`, so the hash of a node is the
   same in every process. Every dataclass becomes
   `@dataclass(frozen=True, eq=False)`, so equality is by identity, which is
   structural because of the table.
1. `dataclasses.replace` and `map_fields` construct through the class, so they
   return interned nodes with no change.

The defaults of `Space` are filled in before the digest is computed, so
`Space("V")` and `Space("V", dual=False)` are the same object.

## Changes by file

| File | Change |
| ---- | ------ |
| `nodes.py` | `Interned`, `encode`, the digest and the table; an `identifier` and `eq=False` on every class |
| `domains.py` | An `identifier` and `eq=False` on every class |
| `static.py` | `by_identity` keyed on the node, or replaced by `functools.cache` with a copy of the dict that `free` returns |
| `oracle.py` | `BUILT` and `VALUES` keyed on nodes in place of `id()` |
| `test/testutil.py` | `contains` keyed on nodes |
| `test/uflx/test_interning.py` | New tests, below |

## Tests

- Equal constructions are the same object, for leaves, nested nodes and
  labels that hold tuples of nodes.
- `Num(1)`, `Num(1.0)`, `Num(True)`, `Num(0.0)`, `Num(-0.0)` and
  `Num(1 + 0j)` are six distinct objects.
- `replace` and `map_fields` with unchanged fields return the same object.
- A node with no other reference is freed, so the table does not grow without
  bound.
- The hash of an expression with `str` labels is the same in two processes
  run with different values of `PYTHONHASHSEED`.
- Moving a class to another module does not change its hash, because the
  hash reads the identifier.
- `density` and `inverse_jacobian` of one map share the object of the metric.
- The whole suite and the negative controls pass, with the time compared to
  4 minutes 47 seconds before the change.
