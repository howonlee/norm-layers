import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

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

def dl1_norm_1d_weird(var, dout):
    norm = np.sum(var)
    var_dout_reduced = np.sum(dout)
    return ((dout * norm) - (var * var_dout_reduced))

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
    init = np.abs(npr.rand(100)) / 10
    init2 = init * 100
    h = np.zeros(100) + 1e-4
    dout = npr.rand(100)

    norm_res = dl1_norm_1d_weird(init, dout)
    norm_res2 = dl1_norm_1d_weird(init2, dout)

    # vjp = putative_vjp(l1_norm_1d, init, h)
    # vjp2 = putative_vjp(l1_norm_1d, init2, h)

    print(init)
    print("=============")
    print("=============")
    print(init2)
    print("=============")
    print("=============")
    print(norm_res)
    print("=============")
    print("=============")
    print(norm_res2)

    plt.plot(norm_res)
    plt.show()
