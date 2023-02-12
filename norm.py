import numpy as np
import numpy.random as npr

# first do L2 norm, then do L1 norm...

def l2_norm(var):
    norm = np.sqrt(np.sum(np.square(var)))
    return var / norm

def dl2_norm(var):
    pass

def putative_vjp(fn, var, h):
    return (fn(var + h) - fn(var - h)) / (2 * h)

if __name__ == "__main__":
    npr.seed(1337)
    init = npr.rand(100)
    h = np.zeros(100) + 1e-3
    print(l2_norm(init))
    print(dl2_norm(init))
    print(putative_vjp(l2_norm, init, h))
