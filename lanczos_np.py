# Lanczos algorithm numpy version.
import numpy as np


def lanczos_numpy(H, order, spec_v):
    """
        H: input matrix, real symmetric, size dim x dim
        order: generate Krylov subspace : span {v, Hv, ..., H^{order}v}
        v: init_vec

        return orthonormal basis of the Krylov subspace of size dim x order
        and the tridiagonal matrix of size order x order.
    """
    float_dtype = np.float64
    dim, _ = H.shape
    tridiag = np.zeros((order, order))
    vecs = np.zeros((dim, order))  # basis of the Krylov subspace
    # init random vec (std. unit Gaussian)
    init_vec = spec_v / np.linalg.norm(spec_v)
    vecs[:, 0:1] = init_vec
    beta = 0
    v_old = np.zeros((dim, 1), dtype=float_dtype)

    for i in range(order):
        v = vecs[:, i:i + 1]
        w = np.matmul(H, v)
        w = w - beta * v_old
        alpha = np.dot(w.ravel(), v.ravel())
        tridiag[i:i+1, i:i+1] = alpha
        w = w - alpha * v

        # Reorthogonalization
        for j in range(i):
            tau = vecs[:, j:j + 1]
            coeff = np.dot(w.ravel(), tau.ravel())
            w = w - coeff * tau
        beta = np.linalg.norm(w)

        if i + 1 < order:
            tridiag[i, i + 1] = beta
            tridiag[i + 1, i] = beta
            vecs[:, i + 1:i + 2] = w / beta
        v_old = v
    return vecs, tridiag


def approx_matrix_fn_times_vec(H, spec_v, fn, order=40):
    """
        Compute f(H) * spec_vec = poly(H) * spec_vec
        H: target matrix
        spec_v: specific vector
        order: polynomial degree that approximates f(x)
    """
    # compute orthonormal basis of Krylov subspace
    V, T = lanczos_numpy(H, order, spec_v)
    # eigen decomp of T
    eigvals, Q = np.linalg.eig(T)
    # compute f(T)
    neweigvals = np.diag(fn(eigvals))
    fT = Q @ neweigvals @ Q.T
    M = V @ fT
    return np.linalg.norm(spec_v) * M[:, 0:1]
