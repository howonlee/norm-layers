import numpy as np
import numpy.random as npr

# first do L2 norm, then do L1 norm...

def l2_norm(var):
    return var / var.sum()

def dl2_norm(var):
    return (var.sum() - (var ** 2)) / (var.sum() ** 2)

if __name__ == "__main__":
    npr.seed(1337)
    init = npr.rand(100)
    h = np.zeros(100) + 1e-3
    print(l2_norm(init))
    print(dl2_norm(init))
    print((l2_norm(init + h) - l2_norm(init - h)) / (2 * h))
