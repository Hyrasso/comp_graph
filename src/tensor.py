from typing import List, Tuple, Union, Iterable, Callable
from functools import reduce, partial
import numpy as np
from functools import reduce
from .F import F, Const

prod = partial(reduce, lambda a, b:a*b)

# def get_shape(l: Union[Tuple, List]):
#     if not all(isinstance(e, (list, tuple)) for e in l):
#         return (len(l),)
#     length = len(l[0])
#     for e in l[1:]:
#         if len(e) != length:
#             return (len(l),)
#     return (len(l),) + get_shape(l[0])

# def flatten(l: Iterable, shape=None):
#     if not shape:
#         return tuple(e for sl in l for e in sl)
#     if len(shape) == 1:
#         return tuple(l)
#     return flatten((li for e in l for li in e), shape=shape[1:])

class Tensor:
    def __init__(self, tensor):
        # self._tensor = flatten(tensor, self.shape)
        self._tensor = np.array(tensor)

    @property
    def shape(self):
        return self._tensor.shape

    # __contains
    # __eq 
    # __getitem
    def __repr__(self):
        return f"Tensor[{','.join(map(str, self.shape))}]"
    
    def __getitem__(self, o):
        res = self._tensor[o]
        if isinstance(res, np.ndarray):
            res = Tensor(res)
        return res
    
    def __setitem__(self, o, value):
        self._tensor[o] = value

    def differentiate(self, var):
        if self == var:
            return Tensor.ones((self.shape))
        return self.apply_each(lambda e:e.differentiate(var))
    
    def apply_each(self, function: Callable):
        self._tensor
        vfunction = np.vectorize(function)
        return Tensor(vfunction(self._tensor))

    def transpose(self):
        return Tensor(self._tensor.T)
    
    @staticmethod
    def ones(shape):
        n = 1
        for d in shape:
            n *= d
        ones = np.array([Const(1)] * n).reshape(shape)
        return Tensor(ones)
    
    @staticmethod
    def from_vector(vector):
        """ Makes a 1xN Tensor from a vector with size N """
        vector = np.array(vector)
        return Tensor(vector.reshape((1, 2)))

def iter_sum(vec: F):
    res = Const(0)
    for e in vec:
        res += e
    return res

def matmul(a, b):
    res = Tensor.ones((a.shape[0], b.shape[1]))
    for i in range(a.shape[1]):
        for j in range(b.shape[0]):
            res[i, j] = iter_sum(a[:, i] * b[j, :])
    return res

