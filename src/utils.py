import numpy as np
import random
import os
import pickle
from itertools import count
from scipy.optimize import minimize


PROJECT_ROOT_DIR = "../../results"
DATA_ID = "../../results/data"
FIGURE_ID = "../../latex/figures"


def image_path(fig_id):
    return os.path.join(FIGURE_ID, fig_id)


def data_path(data_id):
    return os.path.join(DATA_ID, data_id)


def logarithmic_target(params, X, y):
    c = params[0]
    k = params[1]

    loss = np.sum((np.log(1 / (1 - X))**c + k - y)**2)
    return loss


def saver(object, filename):
    pickle.dump(object, open(filename, "wb"))


def loader(filename):
    object = pickle.load(open(filename, "rb"))

    return object


def fit_function(fun, args, x0, method="Nelder-Mead"):
    X, y = args
    X = np.array(X).reshape(1, -1)
    y = np.array(y)

    res = minimize(fun=fun, args=(X, y), x0=x0, method=method)
    c, k = res.x

    return c, k


def kron(*args):
    length = len(args)
    A = args[0]
    for i in range(1, length):
        A = np.kron(A, args[i])

    return A


def agf(U1, U2):
    K = U1.conj().T@U2
    dim = U1.shape[0]
    fid = (np.real(np.trace(K) * np.trace(K.conj().T)) +
           dim) / (dim + dim**2)
    return fid
