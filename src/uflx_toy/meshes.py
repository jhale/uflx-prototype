import numpy as np


def shape_functions(K, X, derivative=False):
    # The degree 1 shape functions of the vertices of K at the points X, in the order of the
    # vertices of Basix: barycentric on a simplex, and products with vertex v at the bits of v.
    X = np.asarray(X, dtype=float)
    k = K.dimension
    if K.num_vertices == k + 1:
        values = np.concatenate([1 - X.sum(axis=-1, keepdims=True), X], axis=-1)
        grads = np.broadcast_to(np.vstack([-np.ones(k), np.eye(k)]), (*X.shape[:-1], k + 1, k))
        return grads if derivative else values
    bits = np.array([[(v >> d) & 1 for d in range(k)] for v in range(K.num_vertices)])
    factors = np.where(bits, X[..., None, :], 1 - X[..., None, :])
    if not derivative:
        return factors.prod(axis=-1)
    grads = np.empty((*X.shape[:-1], K.num_vertices, k))
    for d in range(k):
        others = np.delete(factors, d, axis=-1).prod(axis=-1)
        grads[..., d] = np.where(bits[:, d], 1.0, -1.0) * others
    return grads
