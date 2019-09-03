from typing import List, Tuple, Union, Iterable
from functools import reduce, partial
import numpy as np

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
    def __init__(self, tensor, shape=None):
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
    
    def __matmul__(self, o):
        return self @ o
        

td = (((1, 2), (3, 4), (5, 6))) 
t = Tensor(td)
print(t.shape)
print(t[0, 1])

