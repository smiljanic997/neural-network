import numpy as np
from scipy.special import expit

def sigmoid(x):
        return 1 / (1 + np.exp(-x))

def dsigmoid(x):
        return x * (1 - x)
        # return sigmoid(x) * (1 - sigmoid(x))

def rel(x):
        return max(0, x)

def drel(x):
        if x >= 0:
                return 1
        else:
                return 0