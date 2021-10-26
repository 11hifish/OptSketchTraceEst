import numpy as np
import matplotlib.pyplot as plt
from algorithms.algos import hutchinson, hutchpp, na_hutchpp

# test on synthetic data
n = 100
c = 2

delta = 1e-3
epsilon = 0.01

eigvals = np.array([(i+1)**(-c) for i in range(n)])
A = np.diag(eigvals)

# generate ground truth
lb = 1 - epsilon
ub = 1 + epsilon
gt = np.trace(A)
gt_lb = lb * gt
gt_ub = ub * gt
print('tr(A): {:.6f}, (1-eps)tr(A): {:.6f}, (1+eps)tr(A): {:.6f}'.format(gt, gt_lb, gt_ub))

num_queries = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110])
all_num_failed_trial = np.zeros(len(num_queries))

def mult_trials(A, trials, num_query, algo):
    all_tr_est = np.zeros(trials)
    for t_idx in range(trials):
        estimator = algo(A=A, n_query=num_query)
        all_tr_est[t_idx] = estimator
    failed_trial_idx = np.where((all_tr_est < gt_lb) | (all_tr_est > gt_ub))[0]
    num_failed_trial = len(failed_trial_idx)
    return all_tr_est, num_failed_trial


num_exp_trials = 100

results = np.zeros((3, len(num_queries)))
for idx, nq in enumerate(num_queries):
    print('idx: {}, nq: {}'.format(idx, nq))
    print('starting NA-Hutch++')
    all_tr_est_na, n_failed_na = \
        mult_trials(A, trials=num_exp_trials, num_query=nq, algo=na_hutchpp)
    print('starting Hutch++')
    all_tr_est_hpp, n_failed_hpp = \
        mult_trials(A, trials=num_exp_trials, num_query=nq, algo=hutchpp)
    print('starting Hutchinson')
    all_tr_est_hutch, n_failed_hutch = \
        mult_trials(A, trials=num_exp_trials, num_query=nq, algo=hutchinson)
    results[0, idx] = n_failed_na
    results[1, idx] = n_failed_hpp
    results[2, idx] = n_failed_hutch


# plot # queries vs. failed attempts
algo_names = ['NA-Hutch++',
              'Hutch++',
              'Hutchinson']
for algo_idx, algo_name in enumerate(algo_names):
    plt.plot(num_queries, results[algo_idx], label='{}'.format(algo_name), alpha=0.5)
    plt.scatter(num_queries, results[algo_idx])
    for (x, y) in zip(num_queries, results[algo_idx]):
      plt.annotate(y,
                   (x, y),
                   textcoords="offset points",
                   xytext=(0, 10), ha='center')
plt.title('c = {} # trials = {} \n epsilon = {}'
        .format(c, num_exp_trials, epsilon))
plt.xlabel('# queries')
plt.ylabel('# failed trials')
plt.legend()
plt.grid()
plt.show()

