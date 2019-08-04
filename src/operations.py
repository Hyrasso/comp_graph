""" Define all the python operations and overload them for F base class """
from typing import Any, Tuple
from function import F, Const


@F._add_methods(("__add__", "__radd__"))
class Add(F):
    def __init__(self, a, b):
        self.args = (a, b)
        self.a = a
        self.b = b

    def compute(self) -> Any:
        return self.a.compute() + self.b.compute()

    def grad(self) -> Tuple[int, int]:
        return Const(1), Const(1)

    def __repr__(self) -> str:
        return "(" + repr(self.a) + " + " + repr(self.b) + ")"

    def __add__(self, o):
        if not isinstance(o, F):
            o = Const(o)
        return Add(self, o)

    def __radd__(self, o):
        return Add.__add__(o, self)

@F._add_methods(("__mul__", "__rmul__"))
class Mul(F):
    def __init__(self, a: Any, b: Any):
        self.a = a
        self.b = b
        self.args = (a, b)

    def compute(self) -> Any:
        return self.a.compute() * self.b.compute()

    def grad(self) -> Tuple[Any, Any]:
        return (self.b, self.a)

    def __repr__(self) -> str:
        return repr(self.a) + " * " + repr(self.b)

    def __mul__(self, o):
        if not isinstance(o, F):
            o = Const(o)
        return Mul(self, o)

    def __rmul__(self, o):
        return Mul.__add__(o, self)
