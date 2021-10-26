import numpy as np


def hutchinson_oracle(matvec_fn, n, n_query, G=None):
    """
    Hutchinson's estimator in the mat-vec oracle model.

    Args:
        matvec_fn: matrix-vector multiplication oracle, fn(v) = Av
        n: matrix size
        n_query: # rand vectors
        G: inject rand matrix (for debugging purpose only)

    Returns: estimated trace
    """
    if G is None:
        G = np.random.randn(n, n_query)
    else:
        n_query = G.shape[1]
    AG = np.hstack([matvec_fn(G[:, i:i+1]) for i in range(n_query)])
    return np.trace(G.T @ AG) / n_query


def hutchpp_oracle(matvec_fn, n, n_query, S=None, G=None):
    """
    Hutch ++ in the mat-vec oracle model.

    Args:
        matvec_fn: matrix-vector multiplication oracle, fn(v) = Av
        n: matrix size
        n_query: # rand vectors
        (S, G): inject rand matrix (for debugging purpose only)

    Returns: estimated trace
    """
    if S is None or G is None:
        # split queries
        m = int(n_query / 3)
        g = n_query - 2 * m  # num. queries for small eigenvalues
        S = np.random.randn(n, m)
        G = np.random.randn(n, g)
    else:
        m = S.shape[1]
        g = G.shape[1]
    AS = np.hstack([matvec_fn(S[:, i:i+1]) for i in range(m)])
    Q, _ = np.linalg.qr(AS)
    # trace for large eigenvalues
    AQ = np.hstack([matvec_fn(Q[:, i:i+1]) for i in range(m)])
    large_eig_trace = np.trace(Q.T @ AQ)
    # trace for small eigenvalues
    R = G - Q @ Q.T @ G
    AR = np.hstack([matvec_fn(R[:, i:i+1]) for i in range(g)])
    small_eig_trace = np.trace(R.T @ AR) / g
    return large_eig_trace + small_eig_trace


def na_hutchpp_oracle(matvec_fn, n, n_query, S=None, R=None, G=None):
    """
    NA-Hutch ++ in the mat-vec oracle model.

    Args:
        matvec_fn: matrix-vector multiplication oracle, fn(v) = Av
        n: matrix size
        n_query: # rand vectors
        (S, R, G): inject rand matrix (for debugging purpose only)

    Returns: estimated trace
    """
    if S is None or R is None or G is None:
        s = int(n_query / 4)
        r = int(n_query / 2)
        g = n_query - s - r  # num. queries for small eigenvalues
        S = np.random.randn(n, s)
        R = np.random.randn(n, r)
        G = np.random.randn(n, g)
    else:
        s = S.shape[1]
        r = R.shape[1]
        g = G.shape[1]
    Z = np.hstack([matvec_fn(R[:, i:i+1]) for i in range(r)])  # AR
    W = np.hstack([matvec_fn(S[:, i:i+1]) for i in range(s)])  # AS
    SZ_pinv = np.linalg.pinv(S.T @ Z)
    large_eig_trace = np.trace(SZ_pinv @ W.T @ Z)
    AG = np.hstack([matvec_fn(G[:, i:i+1]) for i in range(g)])
    small_eig_trace = (np.trace(G.T @ AG) - np.trace(G.T @ Z @ SZ_pinv @ W.T @ G)) / g
    return large_eig_trace + small_eig_trace
