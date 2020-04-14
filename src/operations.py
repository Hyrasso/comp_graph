""" Define all the python operations and overload them for Node base class """
from typing import Any, Tuple
from .node import Node
from src.variable import Const
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

@Node.overload_numeric("__add__")
class Add(Node):
    def __init__(self, a: Node, b: Node):
        self.a = a
        self.b = b

    def compute(self) -> Any:
        return self.a.compute() + self.b.compute()

    def grad(self) -> Tuple[Node, Node]:
        return Const(1), Const(1)

    def __str__(self) -> str:
        return f"({self.a} + {self.b})"

    def __add__(self, o):
        if not isinstance(o, Node):
            o = Const(o)
        return Add(self, o)

@Node.overload_numeric("__sub__")
class Sub(Node):
    def __init__(self, a: Node, b: Node):
        super().__init__(a, b)
        self.a = a
        self.b = b

    def compute(self) -> Any:
        return self.a.compute() - self.b.compute()

    def grad(self) -> Tuple[Node, Node]:
        return Const(1), Const(-1)

    def __str__(self) -> str:
        return f"({self.a} - {self.b})"

    def __sub__(self, o):
        if not isinstance(o, Node):
            o = Const(o)
        return Sub(self, o)

@Node.overload_numeric("__neg__")
class Neg(Node):
    def __init__(self, a: Node):
        super().__init__(a)
        self.a = a

    def compute(self) -> Any:
        return - self.a.compute()

    def grad(self) -> Tuple[Node]:
        return (Const(-1),)

    def __str__(self) -> str:
        return f"-{self.a})"

    def __neg__(self):
        return Neg(self)

@Node.overload_numeric("__mul__")
class Mul(Node):
    def __init__(self, a: Node, b: Node):
        self.a = a
        self.b = b

    def compute(self) -> Any:
        return self.a.compute() * self.b.compute()

    def grad(self) -> Tuple[Node, Node]:
        return self.b, self.a

    def __str__(self) -> str:
        return f"({self.a} * {self.b})"

    def __mul__(self, o):
        if not isinstance(o, Node):
            o = Const(o)
        return Mul(self, o)

@Node.overload_numeric("__truediv__")
class Div(Node):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def compute(self) -> Any:
        return self.a.compute() / self.b.compute()

    def grad(self) -> Tuple[Node, Node]:
        return Const(1) / self.b, - self.a / self.b ** 2 

    def __str__(self) -> str:
        return f"({self.a} / {self.b})"

    def __truediv__(self, o):
        if not isinstance(o, Node):
            o = Const(o)
        return Div(self, o)


@Node.overload_numeric("__pow__")
@Node.overload_numeric("__rpow__")
class Pow(Node):
    def __init__(self, a: Node, b: Node):
        super().__init__(a, b)
        self.a = a
        self.b = b

    def compute(self) -> Any:
        return self.a.compute() ** self.b.compute()

    def grad(self) -> Tuple[Node, Node]:
        return self.b * self.a ** (self.b - Const(1)), self.a ** self.b * Log(self.b)

    def __str__(self) -> str:
        return f"({self.a} ^ {self.b})"

    def __pow__(self, o):
        if not isinstance(o, Node):
            o = Const(o)
        return Pow(self, o)

    def __rpow__(self, o):
        if not isinstance(o, Node):
            o = Const(o)
        return Pow(o, self)

# matmul is not implemented for usual python types
# @Node.overload_numeric("__matmul__")
# class Dot(Node):
#     def __init__(self, a: Node, b: Node):
#         super().__init__(a, b)
#         self.a = a
#         self.b = b
#         self.c = self.a @ self.b

#     def compute(self) -> Any:
#         return self.a.compute() @ self.b.compute()

#     def grad(self) -> Tuple[Node, Node]:
#         return self.b, self.a

#     def __str__(self) -> str:
#         return f"({self.a} @ {self.b})"

#     def __matmul__(self, o):
#         if not isinstance(o, Node):
#             o = Const(o)
#         return Dot(self, o)
