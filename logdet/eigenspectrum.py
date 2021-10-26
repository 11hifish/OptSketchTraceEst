import matplotlib.pyplot as plt
import numpy as np
from logdet.data_utils import load_sparse_matrix
import pickle


def load_data(dataname):
    if dataname in ['precipitation']:
        with open('../data/{}.pkl'.format(dataname), 'rb') as f:
            A = pickle.load(f)
    else:
        A = load_sparse_matrix(dataname)
    return A


def get_eigvals(datanames, savename=None):
    rec = {}
    for dname in datanames:
        A = load_data(dname)
        eigvals, eigvecs = np.linalg.eig(A)
        rec[dname] = (eigvals, eigvecs)
    if savename is None:
        savename = 'eigvals'
    with open(savename + '.pkl', 'wb') as f:
        pickle.dump(rec, f)


def plot_eigenspectrum(dataname, normalize=False):
    if dataname in ['precipitation']:
        with open('../data/{}.pkl'.format(dataname), 'rb') as f:
            A = pickle.load(f)
    else:
        A = load_sparse_matrix(dataname)
    eigvals, _ = np.linalg.eig(A)
    if normalize:
        eigvals = eigvals / np.max(eigvals)
    seigvals = np.sort(eigvals)[::-1]
    plt.plot(seigvals)
    plt.title(dataname)
    plt.show()

if __name__ == '__main__':
    datanames = ['bcsstk20', 'bcsstm08']
    get_eigvals(datanames)
