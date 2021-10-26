## A translation of the Matlab code of maxent_logdet.m
## from Entropic Trace Estimataion for Log Determinants
## with different subroutines for trace estimation
## The original code: https://github.com/OxfordML/EntropicTraceEstimation

import numpy as np
from logdet.stable_maxent import stable_maxent


def hutchinson_moment_est(A, dim, n_query, moment):
    """
        Args:
            A: target matrix
            dim: matrix size
            n_query: number of query vectors
            moment: moment

        Returns: estimated moment vector of length moment + 1
    """
    z = np.random.randn(dim, n_query)
    E = np.zeros(moment + 1)  # estimated moment vector for k = {0, 1, 2,..., moment}
    ## trace estimation of matrix moments
    for i in range(n_query):
        # z_ = z[:, i] / np.linalg.norm(z[:, i])
        z_ = z[:, i]
        # start moment estimation from the 3nd moment
        Ez = np.matmul(A, np.matmul(A, z_))
        for j in range(3, moment + 1):
            Ez = np.matmul(A, Ez)
            E[j] = E[j] + np.matmul(z_.T, Ez) / n_query
    E = E / dim
    ## compute accurate moment for k = {0, 1, 2}
    E[0] = 1  # 0-th moment
    E[1] = np.trace(A) / dim  # 1-th moment
    E[2] = np.linalg.norm(A, ord='fro') ** 2 / dim  # 2-nd moment
    return E


def hutchpp_moment_est(A, dim, n_query, moment):
    """
        Args:
            A: target matrix
            dim: matrix size
            n_query: number of query vectors
            moment: moment

        Returns: estimated moment vector of length moment + 1
    """
    E = np.zeros(moment + 1)  # estimated moment vector for k = {0, 1, 2,..., moment}
    s_size = int(n_query / 3)
    g_size = n_query - 2 * s_size
    S = np.random.randn(dim, s_size)
    G = np.random.randn(dim, g_size)

    power_AS = np.matmul(A, np.matmul(A, S))  # A^2 S
    Q, _ = np.linalg.qr(power_AS)
    power_AQ = np.matmul(A, np.matmul(A, Q))  # A^2 Q
    G_proj = G - np.matmul(Q, np.matmul(Q.T, G))
    power_AG_proj = np.matmul(A, np.matmul(A, G_proj))  # A^2(I - QQ.T)G
    for j in range(3, moment + 1):
        power_AS = np.matmul(A, power_AS)
        power_AQ = np.matmul(A, power_AQ)
        power_AG_proj = np.matmul(A, power_AG_proj)
        E[j] = np.trace(Q.T @ power_AQ) + np.trace(G_proj.T @ power_AG_proj) / g_size
    E = E / dim

    ## compute accurate moment for k = {0, 1, 2}
    E[0] = 1  # 0-th moment
    E[1] = np.trace(A) / dim  # 1-th moment
    E[2] = np.linalg.norm(A, ord='fro') ** 2 / dim  # 2-nd moment
    return E


def na_hutchpp_moment_est(A, dim, n_query, moment):
    """
        Args:
            A: target matrix
            dim: matrix size
            n_query: number of query vectors
            moment: moment

        Returns: estimated moment vector of length moment + 1
    """
    E = np.zeros(moment + 1)  # estimated moment vector for k = {0, 1, 2,..., moment}
    s_size = int(n_query / 4)
    r_size = int(n_query / 2)
    g_size = n_query - s_size - r_size  # num. queries for small eigenvalues
    S = np.random.randn(dim, s_size)
    R = np.random.randn(dim, r_size)
    G = np.random.randn(dim, g_size)
    power_AS = np.matmul(A, np.matmul(A, S))  # A^2 S
    power_AR = np.matmul(A, np.matmul(A, R))  # A^2 R
    power_AG = np.matmul(A, np.matmul(A, G))  # A^2 G

    for j in range(3, moment + 1):
        power_AS = np.matmul(A, power_AS)
        power_AR = np.matmul(A, power_AR)
        power_AG = np.matmul(A, power_AG)
        SAR_pinv = np.linalg.pinv(S.T @ power_AR)
        large_eig_trace = np.trace(SAR_pinv @ power_AS.T @ power_AR)
        small_eig_trace = (np.trace(G.T @ power_AG) -
                           np.trace(G.T @ power_AR @ SAR_pinv @ power_AS.T @ G)) / g_size
        E[j] = large_eig_trace + small_eig_trace
    E = E / dim

    ## compute accurate moment for k = {0, 1, 2}
    E[0] = 1  # 0-th moment
    E[1] = np.trace(A) / dim  # 1-th moment
    E[2] = np.linalg.norm(A, ord='fro') ** 2 / dim  # 2-nd moment
    return E


def maxent_logdet_seq(A, moment_est_fn, n_query=30, moment=10, x=None):
    """
        Args:
            A: Input PSD matrix
            moment_est_fn: matrix moment estimation estimation (via trace estimation)
            n_query: number of query vectors for trace estimation

        Returns: An estimation of the log determinant of A
    """
    print('lenA: ', len(A))
    max_e = np.max(np.sum(np.abs(A), axis=1))
    dim = len(A)
    B = A / max_e
    E = moment_est_fn(B, dim, n_query, moment)

    # min_A = 1 / max_e
    min_A = max(np.min(np.diag(B) * 2 - np.sum(np.abs(B), axis=1)), 1e-8)
    if min_A == 1:
        return np.log(max_e) * dim, E
    if x is None:
        x = np.linspace(start=min_A, stop=1, num=int(np.ceil((1 - min_A)/0.01)))
    p, _ = stable_maxent(E, x)
    r = np.dot(p, np.log(x)) / np.sum(p)
    return np.log(max_e) * dim + r * dim, E

