""" Define all the python operations and overload them for F base class """
from typing import Any, Tuple
from .F import F, Const
from functools import reduce
import operator
from .functions import Log

# object.__add__(self, other) - done
# object.__sub__(self, other) - done
# object.__neg__(self) - done
# object.__mul__(self, other) - done
# object.__matmul__(self, other) - need matrix type
# object.__truediv__(self, other) - done
# object.__floordiv__(self, other) - infinite grad for floor
# object.__mod__(self, other) - grad not defined everywhere
# object.__divmod__(self, other) - see mod
# object.__pow__(self, other[, modulo]) - no modulo for you

@F.overload_numeric("__add__")
class Add(F):
    def __init__(self, a: F, b: F):
        self.args = (a, b)
        self.a = a
        self.b = b

    def compute(self) -> Any:
        return self.a.compute() + self.b.compute()

    def grad(self) -> Tuple[F, F]:
        return Const(1), Const(1)

    def __str__(self) -> str:
        return f"({self.a} + {self.b})"

    def __add__(self, o):
        if not isinstance(o, F):
            o = Const(o)
        return Add(self, o)

@F.overload_numeric("__sub__")
class Sub(F):
    def __init__(self, a: F, b: F):
        super().__init__(a, b)
        self.a = a
        self.b = b

    def compute(self) -> Any:
        return self.a.compute() - self.b.compute()

    def grad(self) -> Tuple[F, F]:
        return Const(1), Const(-1)

    def __str__(self) -> str:
        return f"({self.a} - {self.b})"

    def __sub__(self, o):
        if not isinstance(o, F):
            o = Const(o)
        return Sub(self, o)

@F.overload_numeric("__neg__")
class Neg(F):
    def __init__(self, a: F):
        super().__init__(a)
        self.a = a

    def compute(self) -> Any:
        return - self.a.compute()

    def grad(self) -> Tuple[F]:
        return tuple(Const(-1))

    def __str__(self) -> str:
        return f"-{self.a})"

    def __neg__(self):
        return Neg(self)

@F.overload_numeric("__mul__")
class Mul(F):
    def __init__(self, a: F, b: F):
        self.a = a
        self.b = b
        self.args = (a, b)

    def compute(self) -> Any:
        return self.a.compute() * self.b.compute()

    def grad(self) -> Tuple[F, F]:
        return self.b, self.a

    def __str__(self) -> str:
        return f"({self.a} * {self.b})"

    def __mul__(self, o):
        if not isinstance(o, F):
            o = Const(o)
        return Mul(self, o)


@F.overload_numeric("__mul__")
class Mul(F):
    def __init__(self, a: F, b: F):
        self.a = a
        self.b = b
        self.args = (a, b)

    def compute(self) -> Any:
        return self.a.compute() * self.b.compute()

    def grad(self) -> Tuple[F, F]:
        return (self.b, self.a)

    def __str__(self) -> str:
        return f"({self.a} * {self.b})"

    def __mul__(self, o):
        if not isinstance(o, F):
            o = Const(o)
        return Mul(self, o)

@F.overload_numeric("__truediv__")
class Div(F):
    def __init__(self, a, b):
        self.args = (a, b)
        self.a = a
        self.b = b

    def compute(self) -> Any:
        return self.a.compute() / self.b.compute()

    def grad(self) -> Tuple[F, F]:
        return Const(1) / self.b, - self.a / self.b ** 2 

    def __str__(self) -> str:
        return f"({self.a} / {self.b})"

    def __truediv__(self, o):
        if not isinstance(o, F):
            o = Const(o)
        return Div(self, o)


@F.overload_numeric("__pow__")
class Pow(F):
    def __init__(self, a: F, b: F):
        super().__init__(a, b)
        self.a = a
        self.b = b

    def compute(self) -> Any:
        return self.a.compute() ** self.b.compute()

    def grad(self) -> Tuple[F, F]:
        return self.b * self.a ** (self.b - 1), self.a ** self.b * Log(self.b)

    def __str__(self) -> str:
        return f"({self.a} ^ {self.b})"

    def __pow__(self, o):
        if not isinstance(o, F):
            o = Const(o)
        return Pow(self, o)
