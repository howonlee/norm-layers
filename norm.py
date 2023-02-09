import numpy as np
import numpy.random as npr

def norm(var):
    return var / var.sum()

def dnorm(var):
    return (var.sum() - (var ** 2)) / (var.sum() ** 2)

if __name__ == "__main__":
    npr.seed(1337)
    init = npr.rand(100)
    h = np.zeros(100) + 1e-3
    print(norm(init))
    print(dnorm(init))
    print((norm(init + h) - norm(init - h)) / (2 * h))
