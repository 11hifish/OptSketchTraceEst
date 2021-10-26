import numpy as np
from algorithms.algos_oracle import *
from algorithms.algos_parallel import *
import time
import matplotlib.pyplot as plt


algo_names = ['Hutch++', 'NA-Hutch++', 'Hutchinson']
algo_map = {
    algo_names[0]: (hutchpp_oracle, hutchpp_parallel),
    algo_names[1]: (na_hutchpp_oracle, na_hutchpp_parallel),
    algo_names[2]: (hutchinson_oracle, hutchinson_parallel)
}

algo_markers = {
    algo_names[0]: '*',
    algo_names[1]: '^',
    algo_names[2]: 'o'
}


def conduct_one_exp(algo_fn, matvec_fn, n, n_query, gt_lb, gt_ub, n_exp_trial=100):
    num_failure = 0
    total_time = 0
    for t_idx in range(n_exp_trial):
        t1 = time.time()
        est = algo_fn(matvec_fn, n, n_query=n_query)
        t_takes = time.time() - t1
        total_time += t_takes
        # check for failure
        if est < gt_lb or est > gt_ub:
            num_failure += 1
    return num_failure, total_time


def compare_algos(algo_fn, parallel_algo_fn, matvec_fn, n, n_query,
                  gt_lb, gt_ub, n_exp_trial=100):
    nf_algo, time_algo = conduct_one_exp(algo_fn=algo_fn, matvec_fn=matvec_fn, n=n,
                                         n_query=n_query, gt_lb=gt_lb, gt_ub=gt_ub,
                                         n_exp_trial=n_exp_trial)
    print('[Non-parallel] # failure: {}, time: {:.6f}'.format(nf_algo, time_algo))
    nf_parallel, time_parallel = conduct_one_exp(algo_fn=parallel_algo_fn, matvec_fn=matvec_fn, n=n,
                                                 n_query=n_query, gt_lb=gt_lb, gt_ub=gt_ub,
                                                 n_exp_trial=n_exp_trial)
    print('[Parallel] # failure: {}, time: {:.6f}'.format(nf_parallel, time_parallel))
    return nf_algo, time_algo, nf_parallel, time_parallel


def compare_trace_estimation_algos(matvec_fn, n, n_query, gt_lb, gt_ub, n_exp_trial):
    results = {}
    for algo_name in algo_names:
        print('algo_name: {}, n_query: {}'.format(algo_name, n_query))
        algo_fn, parallel_algo_fn = algo_map[algo_name]
        nf_algo, time_algo, nf_parallel, time_parallel = \
            compare_algos(algo_fn, parallel_algo_fn, matvec_fn, n, n_query,
                          gt_lb, gt_ub, n_exp_trial)
        results[algo_name] = nf_algo, time_algo, nf_parallel, time_parallel
    return results


def collect_all_results(all_results):
    num_failures = {}  # algo_name : # failures across queries
    total_time = {}  # algo_name : total time for non-parallel execution
    parallel_time = {}  # algo_name : total time for parallel execution
    for algo_name in algo_names:
        algo_n_failures = np.zeros(len(all_results))
        algo_total_time = np.zeros(len(all_results))
        algo_parallel_time = np.zeros(len(all_results))
        for idx, results in enumerate(all_results):
            nf_algo, time_algo, nf_parallel, time_parallel = results[algo_name]
            algo_n_failures[idx] = nf_algo
            algo_total_time[idx] = time_algo
            algo_parallel_time[idx] = time_parallel
        num_failures[algo_name] = algo_n_failures
        total_time[algo_name] = algo_total_time
        parallel_time[algo_name] = algo_parallel_time
    return num_failures, total_time, parallel_time


def compare_trace_estimation_algos_multi_queries(matvec_fn, n, num_queries,
                                                 gt_lb, gt_ub, n_exp_trial):
    all_results = []
    for n_query in num_queries:
        results = compare_trace_estimation_algos(matvec_fn=matvec_fn, n=n, n_query=n_query,
                                                 gt_lb=gt_lb, gt_ub=gt_ub, n_exp_trial=n_exp_trial)
        all_results.append(results)
    num_failures, total_time, parallel_time = collect_all_results(all_results)
    return num_failures, total_time, parallel_time


### plot
def plot_single(num_queries, res_dic, title=None, ylabel=None, savename=None):
    # plt.figure(figsize=())
    for algo_name in algo_names:
        plt.plot(num_queries, res_dic[algo_name], label=algo_name, alpha=0.5)
        plt.scatter(num_queries, res_dic[algo_name], marker=algo_markers[algo_name])
    plt.legend()
    plt.grid()
    if title is not None:
        plt.title(title)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.xlabel('# queries')
    if savename is not None:
        plt.savefig(savename)
        plt.close()
    else:
        plt.show()


def extract_mean_std(num_queries, list_res_dic):
    mean_dic = {}
    std_dic = {}
    for algo_name in algo_names:
        all_res = np.zeros((len(list_res_dic), len(num_queries)))
        for idx, res_dic in enumerate(list_res_dic):
            single_res = res_dic[algo_name]
            all_res[idx] = single_res
        mean_dic[algo_name] = np.mean(all_res, axis=0)
        std_dic[algo_name] = np.std(all_res, axis=0)
    return mean_dic, std_dic


def plot_multiple(num_queries, list_res_dic, title=None, ylabel=None, savename=None):
    mean_dic = {}
    std_dic = {}
    # preprocess: extract mean and 1 std
    for algo_name in algo_names:
        all_res = np.zeros((len(list_res_dic), len(num_queries)))
        for idx, res_dic in enumerate(list_res_dic):
            single_res = res_dic[algo_name]
            all_res[idx] = single_res
        mean_dic[algo_name] = np.mean(all_res, axis=0)
        std_dic[algo_name] = np.std(all_res, axis=0)
    plt.figure(figsize=(6, 5))
    for algo_name in algo_names:
        plt.errorbar(num_queries, mean_dic[algo_name], yerr=std_dic[algo_name], alpha=0.5,
                     label=algo_name)
        plt.scatter(num_queries, mean_dic[algo_name], marker=algo_markers[algo_name])
    plt.legend()
    plt.grid()
    if title is not None:
        plt.title(title)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.xlabel('# queries')
    if savename is not None:
        plt.savefig(savename, bbox_inches='tight', pad_inches=0.05)
        plt.close()
    else:
        plt.show()
