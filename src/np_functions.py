from src.F import F
import numpy as np


def ctx_from_array(parameters: np.ndarray, values: np.ndarray):
    assert parameters.shape == values.shape, (parameters, values)
    return dict(zip(parameters.flat, values.flat))

def make_value(array: np.ndarray, value=0):
    return np.zeros(array.shape) + value

def gradients(func: F, params: np.ndarray):
    g = [func.differentiate(e) for e in params.flat]
    return np.array(g).reshape(params.shape)

vgradients = np.vectorize(gradients)

def compute(array: np.ndarray):
    res = [e.compute() for e in array.flat]
    return np.array(res).reshape(array.shape)
