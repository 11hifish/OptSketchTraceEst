import numpy as np
import multiprocessing as mp


"""
    Subroutines for parallelism.
"""
def hutchinson_async_fn(matvec_fn, n, vec=None):
    # compute g.T Ag
    if vec is None:
        vec = np.random.randn(n, 1)
    res = vec.T @ matvec_fn(vec)
    return res[0, 0]

def matvec_async_fn(matvec_fn, n, vec=None):
    # compute Ag
    if vec is None:
        vec = np.random.randn(n, 1)
    return matvec_fn(vec)


"""
    3 trace estimation algos.
"""
def hutchinson_parallel(matvec_fn, n, n_query, n_worker=None, G=None):
    """
    Parallel Hutchinson.

    Args:
        matvec_fn: matrix-vector multiplication oracle, fn(v) = Av
        n: matrix size
        n_query: # rand. vecs
        n_worker: # processes
        G: inject rand matrix (for debugging purpose only)

    Returns: estimated trace
    """
    n_worker = mp.cpu_count() if n_worker is None else n_worker
    pool = mp.Pool(n_worker)
    if G is None:
        result_objects = [pool.apply_async(hutchinson_async_fn, args=(matvec_fn, n))
                          for _ in range(n_query)]
    else:
        result_objects = [pool.apply_async(hutchinson_async_fn, args=(matvec_fn, n, G[:, i:i+1]))
                          for i in range(n_query)]
    results = np.array([obj.get() for obj in result_objects])
    pool.close()
    pool.join()
    return np.mean(results)


def hutchpp_parallel(matvec_fn, n, n_query, n_worker=None, S=None, G=None):
    """
    Parallel Hutch++.

    Args:
        matvec_fn: matrix-vector multiplication oracle, fn(v) = Av
        n: matrix size
        n_query: # rand. vecs
        n_worker: # processes
        (S, G): inject rand matrix (for debugging purpose only)

    Returns:estimated trace
    """
    n_worker = mp.cpu_count() if n_worker is None else n_worker
    pool = mp.Pool(n_worker)
    if S is None or G is None:
        # split queries
        m = int(n_query / 3)
        g = n_query - 2 * m  # num. queries for small eigenvalues
        # compute AS via pool
        AS_result_objects = [pool.apply_async(matvec_async_fn, args=(matvec_fn, n))
                             for _ in range(m)]
        AS_res = [obj.get() for obj in AS_result_objects]
        AS = np.hstack(AS_res)
        G = np.random.randn(n, g)
    else:
        m = S.shape[1]
        g = G.shape[1]
        # compute AS via pool
        AS_result_objects = [pool.apply_async(matvec_async_fn, args=(matvec_fn, n, S[:, i:i+1]))
                             for i in range(m)]
        AS_res = [obj.get() for obj in AS_result_objects]
        AS = np.hstack(AS_res)
    # compute Q
    Q, _ = np.linalg.qr(AS)
    R = G - Q @ Q.T @ G
    V = np.hstack([Q, R])
    AV_result_objects = [pool.apply_async(matvec_async_fn, args=(matvec_fn, n, V[:, i:i+1]))
                        for i in range(m+g)]
    AV_res = [obj.get() for obj in AV_result_objects]
    AV = np.hstack(AV_res)
    trace_large = np.trace(Q.T @ AV[:, :m])
    trace_small = np.trace(R.T @ AV[:, m:]) / g
    pool.close()
    pool.join()
    return trace_large + trace_small


def na_hutchpp_parallel(matvec_fn, n, n_query, n_worker=None, S=None, R=None, G=None):
    """
    Parallel NA-Hutch++.

    Args:
        matvec_fn: matrix-vector multiplication oracle, fn(v) = Av
        n: matrix size
        n_query: # rand. vecs
        n_worker: # processes
        (S, R, G): inject rand matrix (for debugging purpose only)

    Returns:estimated trace
    """
    n_worker = mp.cpu_count() if n_worker is None else n_worker
    pool = mp.Pool(n_worker)
    if S is None or R is None or G is None:
        s = int(n_query / 4)
        r = int(n_query / 2)
        g = n_query - s - r
        V = np.random.randn(n, n_query)
    else:
        s = S.shape[1]
        r = R.shape[1]
        g = G.shape[1]
        V = np.hstack([S, R, G])

    # compute all matvec at once via pool
    AV_result_objects = [pool.apply_async(matvec_async_fn, args=(matvec_fn, n, V[:, i:i+1]))
                         for i in range(n_query)]
    AV_res = [obj.get() for obj in AV_result_objects]
    AV = np.hstack(AV_res)
    S, AS = V[:, :s], AV[:, :s]
    AR = AV[:, s:s+r]
    G, AG = V[:, s+r:], AV[:, s+r:]
    SAR_pinv = np.linalg.pinv(S.T @ AR)
    trace_large = np.trace(SAR_pinv @ (AS.T @ AR))
    trace_A = np.trace(G.T @ AG)
    trace_left = np.trace((G.T @ AR) @ SAR_pinv @ (AS.T @ G))
    pool.close()
    pool.join()
    return trace_large + (trace_A - trace_left) / g
