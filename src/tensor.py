from typing import Union, Any
import numpy as np
from src.F import F

# The idea here is to wrap nparray 
# and wrap returns that are ndarray in Tensor 
# this way we can implement the differentiate and compute specifically for arrays, 
# and that'd make the library compatible with numpy

class Tensor:
    @staticmethod
    def __transmute(a:Union[Any, np.ndarray]) -> Union[Any, Tensor]:
        """ Takes any input and return np.ndarray wrapped in Tensors """
        if isinstance(a, np.ndarray):
            return Tensor(a)
        return a

    def __init__(self, *args, **kwargs):
        self._nparray = np.array(*args, **kwargs)

    def __getattr__(self, attr):
        return Tensor.__transmute(getattr(self._nparray, attr))
    
    def __getitem__(self, key):
        return Tensor.__transmute(self._nparray[key])
    
    def __repr__(self):
        return self._nparray.__repr__().replace("array", "Tensor")

    def differentiate(self, other):
        def diff(e):
            return e.differentiate(other)
        if isinstance(other, F):
            return Tensor.__transmute(np.vectorize(diff)(self._nparray))

        if isinstance(other, Tensor):
            return Tensor.gradients(self, other)
        return TypeError(f"Argument should be of type F or Tensor, not {type(other)}")

    @staticmethod
    def gradients(func, params):
        g = [func.differentiate(e) for e in params.flat]
        return Tensor(g).reshape(params.shape)



def ctx_from_array(parameters: np.ndarray, values: np.ndarray):
    assert parameters.shape == values.shape, (parameters, values)
    return dict(zip(parameters.flat, values.flat))

# TODO: remove/rename
def make_value(array: np.ndarray, value=0):
    return np.zeros(array.shape) + value

def gradients(func: F, params: np.ndarray):
    g = [func.differentiate(e) for e in params.flat]
    return np.array(g).reshape(params.shape)

vgradients = np.vectorize(gradients, excluded=("params"),)

def compute(array: np.ndarray):
    res = [e.compute() for e in array.flat]
    return np.array(res).reshape(array.shape)
