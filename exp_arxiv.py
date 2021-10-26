import numpy as np
import pickle
import os
from utils import *


exp_no = 1
with open(os.path.join('data', 'arxiv3.pkl'), 'rb') as f:
    A = pickle.load(f)  # A is in sparse coo_matrix format

# checks
A = A.toarray()
n, _ = A.shape
print(A.shape)
print((A == A.T).all())  # make sure A is symmetric

epsilon = 0.01
# generate ground truth
lb = 1 - epsilon
ub = 1 + epsilon
gt = np.trace(A)
gt_lb = lb * gt
gt_ub = ub * gt
print('tr(A): {:.6f}, (1-eps)tr(A): {:.6f}, (1+eps)tr(A): {:.6f}'.format(gt, gt_lb, gt_ub))


num_queries = np.array([10, 30, 50, 70, 90, 110, 130, 150])
n_exp_trial = 100

def matvec_fn(vec):
    return A @ vec

num_failures, total_time, parallel_time = \
    compare_trace_estimation_algos_multi_queries(matvec_fn=matvec_fn, n=n, num_queries=num_queries,
                                                 gt_lb=gt_lb, gt_ub=gt_ub, n_exp_trial=n_exp_trial)

with open('arxiv_exp_{}.pkl'.format(exp_no), 'wb') as f:
    pickle.dump((num_queries, num_failures, total_time, parallel_time), f)

# # plot # failure across n_exp_trials
# plot_single(num_queries, num_failures,
#             title='# failure across {} trials'.format(n_exp_trial),
#             ylabel='# failure',
#             savename='num_failure_exp_{}'.format(exp_no))
# # plot time for non-parallel execution
# plot_single(num_queries, total_time,
#             title='Total time (non-parallel)',
#             ylabel='seconds',
#             savename='total_time_exp_{}'.format(exp_no))
# # plot time for parallel execution
# plot_single(num_queries, parallel_time,
#             title='Parallel time',
#             ylabel='seconds',
#             savename='parallel_time_exp_{}'.format(exp_no))

