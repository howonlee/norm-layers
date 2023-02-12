import numpy as np
import numpy.random as npr

# first do L2 norm, then do L1 norm...

def l2_norm(var):
    norm = np.sqrt(np.sum(np.square(var)))
    return var / norm

def dl2_norm(var, dout):
    norm = np.sqrt(np.sum(np.square(var)))
    var_dout_reduced = np.sum(var * dout)
    var_dout_reduced /= norm
    return (((dout * norm) - (var * var_dout_reduced)) / (norm ** 2))

def l1_norm(var):
    norm = np.sum(var)
    return var / norm

def dl1_norm(var, dout):
    norm = np.sum(var)
    var_dout_reduced = np.sum(dout)
    return ((dout * norm) - (var * var_dout_reduced)) / (norm ** 2)

def putative_vjp(fn, var, h):
    res = (fn(var + h) - fn(var - h)) / (2 * h)
    return res

if __name__ == "__main__":
    npr.seed(1337)
    init = npr.rand(100)
    h = np.zeros(100) + 1e-3
    print(l2_norm(init))
    print(dl2_norm(init, np.ones(100)))
    print(putative_vjp(l2_norm, init, h))
    print("=============")
    print("=============")
    print("=============")
    
    print(dl1_norm(init, np.ones(100)))
    print(putative_vjp(l1_norm, init, h))
