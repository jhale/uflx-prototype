import sys

import numpy as np

from uflx_spec_interpreter._src.nodes import (
    Absolute,
    ComponentTensor,
    Conjugate,
    Determinant,
    Division,
    Float,
    GatheredIndex,
    Index,
    Indexed,
    IndexSum,
    Integer,
    Inverse,
    Power,
    Product,
    RealPart,
    SquareRoot,
    Sum,
    Tensor,
    Zero,
)
from uflx_toy.nodes import RuntimeArray


def c_kernel(e, name, read):
    # A compiled expression as straight-line C for one cell, with every index unrolled and each
    # entry computed once. read(label, k) is the C expression that reads entry k of the runtime
    # array with the label.
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 100000))
    lines, memo = [], {}

    def temp(code):
        lines.append(f"  const double t{len(lines)} = {code};")
        return f"t{len(lines) - 1}"

    def entry(n, bound, comp):
        key = (n, tuple(bound[i] for i, _ in n.free_indices), comp)
        if key not in memo:
            memo[key] = scalar(n, bound, comp)
        return memo[key]

    def scalar(n, bound, comp):
        match n:
            case Integer(v) | Float(v):
                return float(v)
            case Zero():
                return 0.0
            case RuntimeArray(label, shape):
                return read(label, int(np.ravel_multi_index(comp, shape)) if shape else 0)
            case Sum(a, b):
                return add(entry(a, bound, comp), entry(b, bound, comp))
            case Product(a, b) if a.value_shape:
                return multiply(entry(a, bound, comp), entry(b, bound, ()))
            case Product(a, b):
                return multiply(entry(a, bound, ()), entry(b, bound, comp))
            case Division(a, b):
                return divide(entry(a, bound, comp), entry(b, bound, ()))
            case Power(a, Integer(k)) if k > 0:
                x = entry(a, bound, comp)
                out = x
                for _ in range(k - 1):
                    out = multiply(out, x)
                return out
            case Absolute(a):
                return call("fabs", np.abs, entry(a, bound, comp))
            case SquareRoot(a):
                return call("sqrt", np.sqrt, entry(a, bound, comp))
            case Conjugate(a) | RealPart(a):
                return entry(a, bound, comp)
            case Tensor(entries):
                return entry(entries[comp[0]], bound, comp[1:])
            case Indexed(x, indices):
                return entry(x, bound, tuple(position(k, bound) for k in indices) + comp)
            case ComponentTensor(x, indices):
                inner = bound | dict(zip(indices, comp, strict=False))
                return entry(x, inner, comp[len(indices) :])
            case IndexSum(x, indices):
                extent = dict(x.free_indices)
                total = 0.0
                for values in np.ndindex(*(extent[i] for i in indices)):
                    inner = bound | dict(zip(indices, map(int, values), strict=True))
                    total = add(total, entry(x, inner, comp))
                return total
            case Determinant(a):
                size = a.value_shape[0]
                return determinant(
                    [[entry(a, bound, (i, j)) for j in range(size)] for i in range(size)]
                )
            case Inverse(a) if a.value_shape in ((1, 1), (2, 2), (3, 3)):
                return inverse(a, bound, comp)
        raise TypeError(f"the C backend cannot write {type(n).__name__}")

    def position(k, bound):
        match k:
            case Index():
                return bound[k]
            case GatheredIndex(table, indices):
                values = table.values
                for m in indices:
                    values = values[position(m, bound)]
                return values
        return int(k)

    def inverse(a, bound, comp):
        n = a.value_shape[0]
        A = [[entry(a, bound, (r, c)) for c in range(n)] for r in range(n)]
        det = determinant(A)
        if n == 1:
            return divide(1.0, det)
        i, j = comp
        minor = [[A[r][c] for c in range(n) if c != i] for r in range(n) if r != j]
        cofactor = determinant(minor)
        return divide(cofactor if (i + j) % 2 == 0 else negate(cofactor), det)

    def determinant(A):
        if len(A) == 1:
            return A[0][0]
        if len(A) == 2:
            return add(multiply(A[0][0], A[1][1]), negate(multiply(A[0][1], A[1][0])))
        total = 0.0
        for c in range(3):
            minor = [[A[r][k] for k in range(3) if k != c] for r in (1, 2)]
            term = multiply(A[0][c], determinant(minor))
            total = add(total, term if c % 2 == 0 else negate(term))
        return total

    def add(a, b):
        if isinstance(a, float) and isinstance(b, float):
            return a + b
        if a == 0.0:
            return b
        if b == 0.0:
            return a
        return temp(f"{render(a)} + {render(b)}")

    def multiply(a, b):
        if isinstance(a, float) and isinstance(b, float):
            return a * b
        if a == 0.0 or b == 0.0:
            return 0.0
        if a == 1.0:
            return b
        if b == 1.0:
            return a
        return temp(f"{render(a)} * {render(b)}")

    def divide(a, b):
        if isinstance(a, float) and isinstance(b, float):
            return a / b
        if a == 0.0:
            return 0.0
        return temp(f"{render(a)} / {render(b)}")

    def negate(a):
        return -a if isinstance(a, float) else temp(f"-{a}")

    def call(name, f, a):
        return float(f(a)) if isinstance(a, float) else temp(f"{name}({a})")

    bound = {i: 0 for i, _ in e.free_indices}
    result = entry(e, bound, ())
    body = "\n".join(lines)
    return (
        f"void {name}(double* restrict A, const double* restrict w, const double* restrict c,\n"
        "    const double* restrict coordinate_dofs, const int* restrict entity_local_index,\n"
        "    const unsigned char* restrict quadrature_permutation, void* custom_data)\n"
        f"{{\n{body}\n  A[0] += {render(result)};\n}}\n"
    )


def render(x):
    return repr(x) if isinstance(x, float) else x
