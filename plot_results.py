import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from utils import algo_names, algo_markers, extract_mean_std


exp_nos = np.arange(10) + 1
results_folder = 'results'

data_names = ['synthetic', 'roget', 'arxiv_cm', 'precipitation']
data_res_str = {
    'synthetic': 'synth_exp_',
    'roget': 'roget_exp_',
    'arxiv': 'arxiv_exp_',
    'precipitation': 'precipitation_exp_',
    'arxiv_cm': 'arxiv_cm_exp_'
}

data_spec_results = {}
for data_name in data_names:
    num_queries = None
    num_failures = []
    total_time = []
    parallel_time = []
    for exp_no in exp_nos:
        with open(os.path.join(results_folder, data_name,
                               data_res_str[data_name] + '{}.pkl'.format(exp_no)), 'rb') as f:
            nq, nf, tt, pt = pickle.load(f)
        num_queries = nq
        num_failures.append(nf)
        total_time.append(tt)
        parallel_time.append(pt)
    nf_mean, nf_std = extract_mean_std(num_queries, num_failures)
    tt_mean, tt_std = extract_mean_std(num_queries, total_time)
    pt_mean, pt_std = extract_mean_std(num_queries, parallel_time)
    data_spec_results[data_name] = (num_queries, nf_mean, nf_std, tt_mean, tt_std, pt_mean, pt_std)

target = 'num_failure'
target_position_map = {
    'num_failure': (1, 2),
    'time': (3, 4),
    'parallel_time': (5, 6)
}
target_ylabel_map = {
    'num_failure': '# failed',
    'time': 'seconds',
    'parallel_time': 'seconds'
}
target_title_map = {
    'num_failure': '# failed estimation across 100 trials',
    'time': 'Wall-clock time with sequential execution',
    'parallel_time': 'Wall-clock time with parallel execution'
}

# plot
fontsize = 18
fontsize2 = 25
tick_fontsize = 18
frequency = 2
fig, axes = plt.subplots(1, len(data_names), figsize=(20, 4))
for didx, data_name in enumerate(data_names):
    dres = data_spec_results[data_name]
    nqueries = dres[0]
    idx1, idx2 = target_position_map[target]
    mean_dic = dres[idx1]
    std_dic = dres[idx2]
    # clear x, y ticks
    axes[didx].set_xticks([])
    axes[didx].set_yticks([])
    fig.add_subplot(1, len(data_names), didx+1, frameon=False)
    for algo_name in algo_names:
        plt.errorbar(nqueries, mean_dic[algo_name], yerr=std_dic[algo_name], alpha=0.5,
                     label=algo_name)
        plt.scatter(nqueries, mean_dic[algo_name], marker=algo_markers[algo_name])
    plt.xticks(nqueries[::frequency], nqueries[::frequency], fontsize=tick_fontsize)
    # y_ticks_want = np.array([0, 20, 40, 60, 80, 100])
    y_ticks_default = plt.yticks()[0][1:-1]
    y_ticks_want = np.concatenate((y_ticks_default[::frequency], y_ticks_default[-1:])).astype(np.int32)
    plt.yticks(y_ticks_want, y_ticks_want, fontsize=tick_fontsize)
    # print(y_ticks_want)
    # plt.legend()
    plt.grid()
    plt.title(data_name, fontsize=fontsize2)
fig.add_subplot(111, frameon=False)
plt.tight_layout()
plt.tick_params(labelcolor='none', which='both', pad=15, top=False, bottom=False, left=False, right=False)
plt.xlabel('# queries', fontsize=fontsize2)
plt.ylabel(target_ylabel_map[target], fontsize=fontsize2)
plt.title(target_title_map[target], pad=50, fontsize=fontsize2, fontweight='bold')
# plt.show()
plt.savefig(target, bbox_inches='tight', pad_inches=0.05)
plt.close()
