from __future__ import annotations
from typing import Union, Any, Sequence
import numpy as np
from src.variable import Var, Const
from src.node import Node
from functools import reduce
import operator

# dont wrap, just some helpers functions to deal with it
# TODO: check if arg is a node instead of tensor to call right method
# TODO: export these functions and make them defaults usage of the Var and other nodes methods

def differentiate(tensor, node) -> np.ndarray:
    if isinstance(node, Node):
        return np.vectorize(lambda e:e.differentiate(node))(tensor)
    raise NotImplementedError(f"Dont know what to do with type {type(node)} yet")

def compute(tensor) -> np.ndarray:
    return np.vectorize(lambda e:e.compute())(tensor)

def get_value(tensor) -> np.ndarray:
    """ Works only for tensor of Var """
    return np.vectorize(lambda e:e.value)(tensor)

def set_value(tensor, value):
    def assign(a, b):
        a.value = b
    np.vectorize(assign)(tensor, value)

def var(shape: Union[Sequence[int], int]) -> np.ndarray:
    if isinstance(shape, int):
        shape = (shape,)
    array = np.array([Var() for _ in range(reduce(operator.mul, shape))]).reshape(shape)
    return array

def const(array) -> np.ndarray:
    return np.vectorize(lambda e:Const(e))(array)

def gradients(func: Node, params: np.ndarray):
    g = [func.differentiate(e) for e in params.flat]
    return np.array(g).reshape(params.shape)

# vgradients = np.vectorize(gradients, excluded=("params"),)
