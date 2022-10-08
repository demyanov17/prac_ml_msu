import numpy as np


def grad_finite_diff(function, w, eps=1e-8):

    dim = w.shape[0]
    result = np.zeros(dim)
    for i in range(dim):
        e_i = np.zeros(dim)
        e_i[i] = 1
        result[i] = (function(w + eps * e_i) - function(w)) / eps
    return result