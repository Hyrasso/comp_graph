from __future__ import annotations
from typing import Any, Tuple, Callable
import abc
from functools import wraps
import numpy as np
from collections import ChainMap
from contextlib import contextmanager

# If function: taking comparaison method, operands and a Node  
# turns that into differentiable mathematical expression?
# ex if a > b then foo -> step(a - b) * foo

def set_args(func):
    """ Decorator for __init method, set self._args with *args before calling init """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        self._args = args
        return func(self, *args, **kwargs)
    return wrapper

def no_numpy(func):
    """ Raise TypeError if any argument is a numpy array """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if any(isinstance(e, np.ndarray) for e in args):
            return NotImplementedError()
        return func(*args, **kwargs)
    return wrapper

class F(abc.ABC):
    """Basic block of the computational graph"""

    # type annotation, should be set for each instance
    args: Tuple[Any, ...]

    # global attr (bad)
    context: ChainMap = ChainMap()

    def __init_subclass__(cls):
        if hasattr(cls, "__init__"):

            init = getattr(cls, "__init__")
            setattr(cls, "__init__", set_args(init))
        return super().__init_subclass__()

    # TODO: decorate instead of override, calling super.init from subclass for args unknown by the parent doesnt make much sense
    def __init__(self, *args, **kwargs):
        self._args = args

    @abc.abstractmethod
    def compute(self) -> Any:
        ...

    @staticmethod
    def overload_numeric(method_name: str) -> F:
        """ Add methods from class to F, used to overload __operations """
        def decorator(cls: F) -> F:
            method = getattr(cls, method_name)
            method = no_numpy(method)
            setattr(F, method_name, method)
            return cls
        return decorator

    def differentiate(self, var: F) -> F:
        """ Define partial derivative relative to var """
        # chain rule for multi variable
        # z = f + g
        # f can be const, variable, functions of anything
        # dz/dx = dz/df * df/dx + dz/dg * dg/dx
        res = None
        for arg, darg in zip(self._args, self.grad()):
            # recursive fonction, skip for now because of how Var is defined
            if self == arg:
                continue
            dz_df = darg
            df_dvar = arg.differentiate(var)
            if res is not None:
                res = res + dz_df * df_dvar
            else:
                res = dz_df * df_dvar
        # print ("derivate", self, "with relate to", var)
        # print("res", res)
        if res is None:
            return Const(0)
        return res

    def __repr__(self) -> str:
        return "{}({})".format(self.__class__.__name__, ", ".join(map(repr, self._args)))
    
    @contextmanager
    def set_context(self, ctx):
        F.context = F.context.new_child(ctx)
        yield
        F.context = F.context.parents

    def __eq__(self, value):
        if isinstance(value, F):
            return (type(self), self._args) == (type(value), value._args)
        return super().__eq__(value)

    def __hash__(self):
        return super().__hash__()

    def __call__(self, ctx) -> F:
        """Replace all var in ctx with their value given, leave the other unchanged return F"""
        self.set_var_val()

class Const(F):
    def __init__(self, a: Any):
        self.a = a

    def compute(self) -> Any:
        return self.a

    def grad(self) -> Tuple[int]:
        return (Const(0),)
    
    def differentiate(self, var: F) -> F:
        return Const(0)

    def __repr__(self) -> str:
        return repr(self.a)
