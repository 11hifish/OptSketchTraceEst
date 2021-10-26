import numpy as np


def hutchinson(A, n_query, G=None):
    """
    Hutchinson's estimator.

    Args:
        A: target symmetric matrix, size n x n
        n_query: # rand vectors
        G: inject rand matrix (for debugging purpose only)

    Returns: estimated trace
    """
    n, _ = A.shape
    if G is None:
        G = np.random.randn(n, n_query)
    return np.trace(G.T @ A @ G) / n_query


def hutchpp(A, n_query, S=None, G=None):
    """
    Hutch ++.

    Args:
        A: target PSD matrix, size n x n
        n_query: # rand vectors
        (S, G): inject rand matrix (for debugging purpose only)

    Returns: estimated trace
    """
    n, _ = A.shape
    if S is None or G is None:
        # split queries
        m = int(n_query / 3)
        g = n_query - 2 * m  # num. queries for small eigenvalues
        S = np.random.randn(n, m)
        G = np.random.randn(n, g)
    else:
        g = G.shape[1]
    Q, _ = np.linalg.qr(A @ S)
    large_eig_trace = np.trace(Q.T @ A @ Q)
    P = np.eye(n) - Q @ Q.T
    small_eig_trace = np.trace(G.T @ P @ A @ P @ G) / g
    return large_eig_trace + small_eig_trace


def na_hutchpp(A, n_query, S=None, R=None, G=None):
    """
    NA-Hutch ++.

    Args:
        A: target PSD matrix, size n x n
        n_query: # rand vectors
        (S, R, G): inject rand matrix (for debugging purpose only)

    Returns: estimated trace
    """
    n, _ = A.shape
    if S is None or R is None or G is None:
        s = int(n_query / 4)
        r = int(n_query / 2)
        g = n_query - s - r  # num. queries for small eigenvalues
        S = np.random.randn(n, s)
        R = np.random.randn(n, r)
        G = np.random.randn(n, g)
    else:
        g = G.shape[1]
    Z = A @ R
    W = A @ S
    SZ_pinv = np.linalg.pinv(S.T @ Z)
    large_eig_trace = np.trace(SZ_pinv @ W.T @ Z)
    small_eig_trace = (np.trace(G.T @ A @ G) - np.trace(G.T @ Z @ SZ_pinv @ W.T @ G)) / g
    return large_eig_trace + small_eig_trace
