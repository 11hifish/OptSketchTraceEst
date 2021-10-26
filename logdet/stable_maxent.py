## A translation of the Matlab code of stableMaxEnt.m
## from Entropic Trace Estimataion for Log Determinants
## The original code: https://github.com/OxfordML/EntropicTraceEstimation
import numpy as np

def stable_maxent(mu, x):
    """

    Args:
        mu: Moment information, vector with {0, 1, 2, ...}-th moment estimation
        x: Sample points used

    Returns:
        n: coefficients that match moments
        p: distribution of eigenvalues at sampled points

    """
    n = np.zeros(len(mu))
    A = np.ones((len(x), len(mu)))  # samples x moments
    for i in range(1, len(mu)):
        A[:, i] = x ** i
    for k in range(1, 200000+1):  # arbitrary loop length
        p_tilde = np.exp(np.matmul(A, n) - 1)
        i = k % len(mu)
        lda = np.log(mu[i] / np.dot(A[:, i], p_tilde))
        n[i] = n[i] + lda
    p = np.exp(np.matmul(A, n) - 1)
    return p, n

