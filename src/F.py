from __future__ import annotations
from typing import Any, Tuple, Callable
import abc

# For 'recursive eval", set context, propage it to all children
# add recursive eval where call eval for children first with context

# For now use evaluate fonction
#  in the future move everything to other some 'compile' method
#  that can generate 'optimized' code for evalutaion

# If function: taking comparaison method, operands and a Node  
# turns that into differentiable mathematical expression?
# ex if a > b then foo -> step(a - b) * foo
class F(abc.ABC):
    """Basic block of the computational graph"""

    args: Tuple[Any, ...] = tuple()
    grad: Tuple[F, ...] = tuple()

    def __init_subclass__(cls):
        if hasattr(cls, "compute"):
            _compute = getattr(cls, "compute")
            def compute(self, ctx=None):
                if ctx:
                    self.set_context(ctx)
                return _compute(self)
            setattr(cls, "compute", compute)
        return super().__init_subclass__()
    def __init__(self, *args, **kwargs):
        self.args = tuple(args)

    @abc.abstractmethod
    def compute(self) -> Any:
        return NotImplemented

    @staticmethod
    def overload_numeric(method: Callable[[Any], F]) -> Callable[[F], F]:
        """ Add methods from class to F, used to overload __operations """
        def decorator(cls: F) -> F:
            setattr(F, method, getattr(cls, method))

            return cls
        return decorator

    def differentiate(self, var: F) -> F:
        """ Define partial derivative relative to var
            TODO: add wrt tensor
        """
        # chain rule for multi variable
        # z = f + g
        # f can be const, variable, functions of anything
        # dz/dx = dz/df * df/dx + dz/dg * dg/dx
        res = None
        for arg, darg in zip(self.args, self.grad()):
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
        return "{}({})".format(self.__class__.__name__, ", ".join(map(repr, self.args)))

    # find a better way,
    # if a node is in 2 graphs, its context will be changed by both
    # now set_context needs to be used before every call to compute
    # ideally have a function to get a function from the graph,
    # give all vars to that function, returns result
    def set_context(self, ctx):
        assert isinstance(ctx, dict), (self, ctx)
        if self in ctx:
            self.value = ctx[self]
        for arg in self.args:
            arg.set_context(ctx)

    def __eq__(self, value):
        if isinstance(value, F):
            return (type(self), self.args) == (type(value), value.args)
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
