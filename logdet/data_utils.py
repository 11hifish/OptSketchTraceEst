from scipy.io import loadmat
import numpy as np
import os


def load_sparse_matrix(dataname):
    data_path = os.path.join('..', 'data', dataname+'.mat')
    data = loadmat(data_path)
    X_sparse = data['Problem']['A'][0, 0]
    X = X_sparse.toarray()
    return X
