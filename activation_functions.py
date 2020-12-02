import numpy as np

def relu(x):
    return np.where(x<0, 0, x)

def neg_relu(x):
    return np.where(x>0, 0, x)

def tanh(x):
    return np.tanh(x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def none(x):
    return x