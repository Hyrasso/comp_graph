from __future__ import annotations
from typing import Any, Tuple, Callable, Type, Dict
import abc
from functools import wraps
import numpy as np
from collections import ChainMap
from contextlib import contextmanager

# If function: taking comparaison method, operands and a Node  
# turns that into differentiable mathematical expression?
# ex if a > b then foo -> step(a - b) * foo

def set_args(func):
    """ Decorator for __init method, set self.args with *args before calling init """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        self.args = args
        return func(self, *args, **kwargs)
    return wrapper

class Node(abc.ABC):
    """Basic block of the computational graph"""

    # must be set for each instance
    args: Tuple[Node, ...]

    def __init_subclass__(cls) -> None:
        if hasattr(cls, "__init__"):

            init = getattr(cls, "__init__")
            setattr(cls, "__init__", set_args(init))
        return super().__init_subclass__()

    def __init__(self, *args, **kwargs) -> None:
        self.args = args

    @abc.abstractmethod
    def compute(self) -> Any:
        ...
    
    @abc.abstractmethod
    def grad(self) -> Any:
        ...

    @staticmethod
    def overload_numeric(method_name: str) -> Callable[[Type[Node]], Type[Node]]:
        """ Add methods from class to Node, used to overload dunder operations methods """
        def decorator(cls: Type[Node]) -> Type[Node]:
            method = getattr(cls, method_name)
            setattr(Node, method_name, method)
            return cls
        return decorator

    def differentiate(self, var: Node) -> Node:
        """ Define partial derivative relative to var """
        # chain rule for multi variable
        # z = f + g
        # f can be const, variable, functions of anything
        # dz/dx = dz/df * df/dx + dz/dg * dg/dx
        if var is None:var = self
        res = None
        for arg, darg in zip(self.args, self.grad()):
            # recursive fonction, skip because of how Var is defined
            # if self == arg:
            #     continue
            dz_df = darg
            df_dvar = arg.differentiate(var)
            if res is not None:
                res = res + dz_df * df_dvar
            else:
                res = dz_df * df_dvar
        if res is None:
            return Const(0)
        return res

    def __repr__(self) -> str:
        return "{}({})".format(self.__class__.__name__, ", ".join(map(repr, self.args)))
    
    @contextmanager
    def set_context(self, ctx: Dict=None):
        if ctx is None:ctx = {}
        for var, value in ctx.items():
            ctx[var] = var.value
            var.value = value
        yield
        for var, value in ctx.items():
            var.value = value

    def __eq__(self, value):
        if isinstance(value, Node):
            return (type(self), self.args) == (type(value), value.args)
        return super().__eq__(value)

    def __hash__(self):
        return super().__hash__()

    def __call__(self, ctx: Dict=None) -> Node:
        """Returns self.compute() with ctx as context  
            r = f({f:1})
            # is equivalent to
            with f.set_context({f:1}):
                tmp = f.compute()
            r = tmp
        """
        with self.set_context(ctx):
            return self.compute()
