from __future__ import annotations
from typing import Any, Tuple, Callable


class SubclassError(Exception):
    pass


# For 'recursive eval", set context, propage it to all children
# add recursive eval where call eval for children first with context

# For now use evaluate fonction
#  in the future move everything to other some 'compile' method
#  that can generate 'optimized' code for evalutaion

# If function taking comparaison method, operands and a Node  
# turns that into differentiable mathematical expression?
# ex if a > b then foo -> step(a - b) * foo
class F:
    """Basic block of the computational graph"""

    args: Tuple[Any, ...]

    def __init_subclass__(cls: Any, **kwargs):
        """ Check for subclass validity.

        """

        if not hasattr(cls, 'compute'):
            raise SubclassError(
                f"Class '{cls.__name__}' doesn't have an 'compute' mehtod")
        if not hasattr(cls, 'differentiate'):
            raise SubclassError(
                f"Class '{cls.__name__}' doesn't have a 'differentiate' mehtod")

    @staticmethod
    def _add_methods(methods: Tuple[str, ...]) -> Callable[[F], F]:
        """ Add methods from class to F, used to overload __operations """
        def decorator(cls: F) -> F:
            for method in methods:
                setattr(F, method, getattr(cls, method))
            return cls
        return decorator

    def differentiate(self, var: F) -> F:
        """ Define partial derivative relative to var """
        # chain rule for multi variable
        # z = f + g
        # f can be const, variable, functions of anything
        # dz/dx = dz/df * df/dx + dz/dg * dg/dx
        O = Const(0)
        res = O
        for i, (arg, darg) in enumerate(zip(self.args, self.grad())):
            # recursive fonction, skip for now because of how Var is defined
            if self == arg:
                continue
            dz_df = darg
            df_dvar = arg.differentiate(var)
            if res is not O:
                res = res + dz_df * df_dvar
            else:
                res = dz_df * df_dvar
        # print ("derivate", self, "with relate to", var)
        # print("res", res)
        return res

    def __repr__(self) -> str:
        return "{}({})".format(self.__class__.__name__, " ".join(map(repr, self.args)))

    def set_context(self, ctx):
        self.ctx = ctx
        for arg in self.args:
            arg.set_context(ctx)

class Const(F):
    def __init__(self, a: Any):
        self.args = tuple()
        self.a = a

    def compute(self) -> Any:
        return self.a

    def grad(self) -> tuple[int]:
        return (Const(0),)
    
    def differentiate(self, var: F) -> F:
        return Const(0)

    def __repr__(self) -> str:
        return repr(self.a)


class Var(F):
    def __init__(self, name: str = "Var"):
        self.name = name
        self.args = tuple()

    def compute(self) -> F:
        return self.ctx[self]

    def grad(self):
        return (Const(1),)

    def __repr__(self):
        return self.name
    
    def differentiate(self, var):
        return Const(1) if self == var else Const(0)

