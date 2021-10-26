import pickle
import numpy as np
from logdet.maxent_logdet_seq import *
from logdet.data_utils import load_sparse_matrix
import matplotlib.pyplot as plt
import multiprocessing as mp


def compare_varying_moments(dataname='synthetic'):
    moments = np.array([10, 15, 20, 25, 30])

    if dataname == 'synthetic':
        n = 1000
        c = 2
        eigvals = np.array([(i + 1) ** (-c) for i in range(n)])
        A = np.diag(eigvals)
    else:
        A = load_sparse_matrix(dataname)
        eigvals, _ = np.linalg.eig(A)
    gt = np.sum(np.log(eigvals))
    n_query = 30
    trial_num = 100

    all_logdet_seq = np.zeros((len(moments), trial_num))
    all_logdet_pp = np.zeros((len(moments), trial_num))
    all_logdet_na = np.zeros((len(moments), trial_num))

    for m_idx, moment in enumerate(moments):
        print('moment: ', moment)
        pool = mp.Pool(mp.cpu_count())
        hut_objects = [pool.apply_async(maxent_logdet_seq, args=(A, hutchinson_moment_est, n_query, moment))
                       for _ in range(trial_num)]
        hut_res = np.array([obj.get()[0] for obj in hut_objects])
        all_logdet_seq[m_idx] = hut_res

        pp_objects = [pool.apply_async(maxent_logdet_seq, args=(A, hutchpp_moment_est, n_query, moment))
                       for _ in range(trial_num)]
        pp_res = np.array([obj.get()[0] for obj in pp_objects])
        all_logdet_pp[m_idx] = pp_res

        na_objects = [pool.apply_async(maxent_logdet_seq, args=(A, na_hutchpp_moment_est, n_query, moment))
                      for _ in range(trial_num)]
        na_res = np.array([obj.get()[0] for obj in na_objects])
        all_logdet_na[m_idx] = na_res

        pool.close()
        pool.join()


    # save results
    with open('results_{}.pkl'.format(dataname), 'wb') as f:
        pickle.dump((all_logdet_seq, all_logdet_pp, all_logdet_na,
                     moments, gt), f)

    # get mean
    all_mean_logdet_seq = np.mean(all_logdet_seq, axis=1)
    all_mean_logdet_pp = np.mean(all_logdet_pp, axis=1)
    all_mean_logdet_na = np.mean(all_logdet_na, axis=1)
    # plot results
    plt.scatter(moments, np.abs(all_mean_logdet_seq - gt) / np.abs(gt), label='seq')

    plt.scatter(moments, np.abs(all_mean_logdet_pp - gt) / np.abs(gt), label='pp')
    plt.scatter(moments, np.abs(all_mean_logdet_na - gt) / np.abs(gt), label='na')

    plt.legend()
    plt.savefig('plot_{}.png'.format(dataname))
    plt.close()


if __name__ == '__main__':
    dataname = 'bcsstk20'
    compare_varying_moments(dataname)
