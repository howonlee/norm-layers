import numpy as np
import numpy.random as npr

def l2_norm(var):
    norm = np.sqrt(np.sum(np.square(var)))
    return var / norm

def dl2_norm(var, dout):
    norm = np.sqrt(np.sum(np.square(var)))
    var_dout_reduced = np.sum(var * dout)
    var_dout_reduced /= norm
    return (((dout * norm) - (var * var_dout_reduced)) / (norm ** 2))

def l1_norm_1d(var):
    norm = np.sum(var)
    return var / norm

def dl1_norm_1d(var, dout):
    norm = np.sum(var)
    var_dout_reduced = np.sum(dout)
    return ((dout * norm) - (var * var_dout_reduced)) / (norm ** 2)

def l1_norm_2d(var):
    norm = np.sum(var, axis=0)
    return var / norm

def dl1_norm_2d(var, dout):
    norm = np.sum(var, axis=0)
    var_dout_reduced = np.sum(dout, axis=0)
    return ((dout * norm) - (var * var_dout_reduced)) / (norm ** 2)

def putative_vjp(fn, var, h):
    res = (fn(var + h) - fn(var - h)) / (2 * h)
    return res

def cos(var):
    return np.cos(var)

def dcos(var, dout):
    return dout * -np.sin(var)

if __name__ == "__main__":
    npr.seed(1337)
    init = npr.rand(100) * 1000
    h = np.zeros(100) + 1e-4
    dout = np.ones(100)

    norm_res = dl1_norm_1d(init, dout)

    vjp = putative_vjp(l1_norm_1d, init, h)

    print(norm_res)
    print("=============")
    print("=============")
    print(vjp)
