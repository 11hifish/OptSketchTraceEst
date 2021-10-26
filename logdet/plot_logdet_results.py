import matplotlib.pyplot as plt
import pickle
import numpy as np
import os


def plot_single_res(res_file):
    with open(res_file, 'rb') as f:
        all_logdet_seq, all_logdet_pp, all_logdet_na, \
        moments, gt = pickle.load(f)
    dataname = res_file.split('.')[0]
    dataname = dataname.split('_')[1]

    # get mean
    all_mean_logdet_seq = np.mean(all_logdet_seq, axis=1)
    all_mean_logdet_pp = np.mean(all_logdet_pp, axis=1)
    all_mean_logdet_na = np.mean(all_logdet_na, axis=1)
    # plot results
    plt.scatter(moments, np.abs(all_mean_logdet_seq - gt) / np.abs(gt), label='seq')

    plt.scatter(moments, np.abs(all_mean_logdet_pp - gt) / np.abs(gt), label='pp')
    plt.scatter(moments, np.abs(all_mean_logdet_na - gt) / np.abs(gt), label='na')

    plt.legend()
    plt.title(dataname)
    plt.show()

def plot_multiple_results(datanames, figsize=(10, 3)):
    fig, axes = plt.subplots(1, len(datanames), figsize=figsize)
    alpha = 0.8
    for didx, dname in enumerate(datanames):
        dpath = os.path.join('results', 'results_{}.pkl'.format(dname))
        with open(dpath, 'rb') as f:
            all_logdet_seq, all_logdet_pp, all_logdet_na, \
            moments, gt = pickle.load(f)

        # get mean
        all_mean_logdet_seq = np.mean(all_logdet_seq, axis=1)
        all_mean_logdet_pp = np.mean(all_logdet_pp, axis=1)
        all_mean_logdet_na = np.mean(all_logdet_na, axis=1)

        # clear x, y ticks
        axes[didx].set_xticks([])
        axes[didx].set_yticks([])
        fig.add_subplot(1, len(datanames), didx + 1, frameon=False)
        # plot results
        plt.scatter(moments, np.abs(all_mean_logdet_seq - gt) / np.abs(gt), label='Hutchinson', alpha=alpha)
        plt.scatter(moments, np.abs(all_mean_logdet_pp - gt) / np.abs(gt), label='Hutch++', alpha=alpha)
        plt.scatter(moments, np.abs(all_mean_logdet_na - gt) / np.abs(gt), label='NA-Hutch++', alpha=alpha)
        plt.title(dname)
        plt.legend()
        plt.grid(axis='x')

    fig.add_subplot(111, frameon=False)
    plt.tight_layout()
    plt.tick_params(labelcolor='none', which='both', pad=15, top=False, bottom=False, left=False, right=False)
    fontsize2 = 15
    plt.xlabel('Moment', fontsize=fontsize2)
    plt.ylabel('Absolute Relative Error', fontsize=fontsize2)
    # plt.title('Entropic Log Determinant Estimation', pad=50, fontsize=fontsize2, fontweight='bold')
    plt.show()

def plot_eigenspectrum(datanames, eigval_path, figsize=(10, 3)):
    with open(eigval_path, 'rb') as f:
        eigval_dic = pickle.load(f)
    fig, axes = plt.subplots(1, len(datanames), figsize=figsize)
    for didx, dname in enumerate(datanames):
        eigvals = eigval_dic[dname]
        eigvals = np.sort(eigvals)[::-1]

        # clear x, y ticks
        axes[didx].set_xticks([])
        axes[didx].set_yticks([])
        fig.add_subplot(1, len(datanames), didx + 1, frameon=False)
        # plot eigenspectrum
        plt.scatter(np.arange(len(eigvals)), eigvals, s=5)
        plt.title(dname)
    fig.add_subplot(111, frameon=False)
    plt.tight_layout()
    plt.tick_params(labelcolor='none', which='both', pad=15, top=False, bottom=False, left=False, right=False)
    fontsize2 = 15
    plt.xlabel('Eigenspectrum', fontsize=fontsize2)
    plt.ylabel('Eigenvalues', fontsize=fontsize2)
    # plt.title('Entropic Log Determinant Estimation', pad=50, fontsize=fontsize2, fontweight='bold')
    plt.show()


def print_table_results(datanames):
    for dname in datanames:
        print('Data: {}'.format(dname))
        res_path = os.path.join('results', 'results_{}.pkl'.format(dname))
        with open(res_path, 'rb') as f:
            all_logdet_seq, all_logdet_pp, all_logdet_na, \
            moments, gt = pickle.load(f)
        # get mean
        all_mean_logdet_seq = np.mean(all_logdet_seq, axis=1)
        all_mean_logdet_pp = np.mean(all_logdet_pp, axis=1)
        all_mean_logdet_na = np.mean(all_logdet_na, axis=1)
        # get abs dev
        absdev_hut = np.abs(all_mean_logdet_seq - gt) / np.abs(gt)
        absdev_pp = np.abs(all_mean_logdet_pp - gt) / np.abs(gt)
        absdev_na = np.abs(all_mean_logdet_na - gt) / np.abs(gt)

        for midx, moment in enumerate(moments):
            print('Moment: {}'.format(moment))
            this_moment_res_hut = absdev_hut[midx]
            this_moment_res_pp = absdev_pp[midx]
            this_moment_res_na = absdev_na[midx]
            print('{:.4f} & {:.4f} & {:.4f}'
                  .format(this_moment_res_hut, this_moment_res_pp, this_moment_res_na))
        print('=========')


if __name__ == '__main__':
    # res_file = 'results/results_bcsstk20.pkl
    # res_file = 'results/results_bcsstm08.pkl'
    # plot_single_res(res_file)
    datanames = ['bcsstk20', 'bcsstm08']
    plot_multiple_results(datanames, figsize=(8, 3))
    plot_eigenspectrum(datanames, eigval_path='eigvals.pkl', figsize=(8, 3))
    # print_table_results(datanames)

