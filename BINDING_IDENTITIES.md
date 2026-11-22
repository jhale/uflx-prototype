# Generated names and structural identity

Generated names are structural data. Their purpose, dependencies, order and tuple
nesting determine identity and therefore interning, sharing and cache reuse.
Refactoring their construction must preserve these data, not merely numerical
results. The internal helpers live in `src/uflx/_src/names.py`.

## Derived identities

`derived_index(purpose, *dependencies)` constructs exactly
`Index((purpose, *dependencies))`. `derived_indices(purpose, operands, n)` constructs the
same labels with an axis position appended for each of `n` indices. Neither is
a counter or an unused-name search. Callers retain their existing dependencies.

Substitution rewrites nodes within derived labels. For example, replacing `A`
with `B` in the indices of `transpose(A)` produces the indices of `transpose(B)`.
This lets independently constructed expressions converge to the same interned
object. Cell-family and quadrature-rule identities use this mechanism too; their
dependencies deliberately identify shared data across expressions.

## Capture-avoidance identities

Core substitution sometimes needs to rename a binder before inserting an
expression that contains its name freely. `renamed_label(node, name, avoid)`
records exactly the existing tuple:

```python
("uflx.rename", name, scoped_operands, tuple(avoid))
```

It refers to the scoped operands, not the binding node itself. The original name
and ordered avoidance context participate in identity. `is_renamed_label`
recognises these labels in one place. Once created, they stay fixed when
substitution encounters their contents. Direct replacement of the named object
still takes precedence. Compiler index instantiation retains its separate
`compiler.simplify` derived identities and existing traversal behaviour.

The scope traversal, capture checks, avoidance lists and substitution ordering
are unchanged. In particular, this refactor does not sort the avoidance context,
drop label dependencies, or replace dependencies with digests. Existing labels
still retain their dependency graphs.
