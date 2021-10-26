import numpy as np
import pickle
import os
from lanczos_np import approx_matrix_fn_times_vec
from utils import *


exp_no = 1
# get Roget data
with open(os.path.join('data', 'roget.pkl'), 'rb') as f:
    A = pickle.load(f)

# checks
n, _ = A.shape
print('A shape: {}'.format(A.shape))
print((A == A.T).all())  # make sure A is symmetric

epsilon = 0.01
# generate ground truth
lb = 1 - epsilon
ub = 1 + epsilon
eigvals, _ = np.linalg.eigh(A)
gt = np.sum(np.exp(eigvals))
gt_lb = lb * gt
gt_ub = ub * gt
print('tr(A): {:.6f}, (1-eps)tr(A): {:.6f}, (1+eps)tr(A): {:.6f}'.format(gt, gt_lb, gt_ub))

def matvec_fn(vec):
    # Run Lanczos for 40 steps to approx exp(A) * v
    res = approx_matrix_fn_times_vec(H=A, spec_v=vec, fn=np.exp, order=40)
    return res

num_queries = np.array([10, 30, 50, 70, 90, 110, 130, 150])
n_exp_trial = 100


num_failures, total_time, parallel_time = \
    compare_trace_estimation_algos_multi_queries(matvec_fn=matvec_fn, n=n, num_queries=num_queries,
                                             gt_lb=gt_lb, gt_ub=gt_ub, n_exp_trial=n_exp_trial)

with open('roget_exp_{}.pkl'.format(exp_no), 'wb') as f:
    pickle.dump((num_queries, num_failures, total_time, parallel_time), f)

# plot # failure across n_exp_trials
plot_single(num_queries, num_failures,
            title='# failure across {} trials'.format(n_exp_trial),
            ylabel='# failure',
            savename='num_failure_exp_{}'.format(exp_no))
# plot time for non-parallel execution
plot_single(num_queries, total_time,
            title='Total time (non-parallel)',
            ylabel='seconds',
            savename='total_time_exp_{}'.format(exp_no))
# plot time for parallel execution
plot_single(num_queries, parallel_time,
            title='Parallel time',
            ylabel='seconds',
            savename='parallel_time_exp_{}'.format(exp_no))
