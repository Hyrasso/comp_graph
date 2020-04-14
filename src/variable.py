from __future__ import annotations
from typing import Any, Tuple, Optional
from .node import Node

class VarError(RuntimeError):
    ...

class Var(Node):
    def __init__(self, name: str = "Var", value: Any = None):
        self.name = name
        self.value = value if value is not None else self
 
    def compute(self) -> Any:
        return self.value

    def grad(self) -> Tuple[Const,]:
        return (Const(1),)

    def __repr__(self):
        return self.name
    
    def differentiate(self, var):
        return Const(1) if self == var else Const(0)

    # overload Node.__eq__, var is only equal to itself
    def __eq__(self, value):
        return self is value
    
    # if eq is defined, hash must be too?
    def __hash__(self):
        return super().__hash__()


class Const(Node):
    def __init__(self, value: Any):
        self.value = value

    def compute(self) -> Any:
        return self.value

    def grad(self) -> Tuple[Const,]:
        return (Const(0),)
    
    def differentiate(self, var: Node) -> Const:
        return Const(0)

    def __repr__(self) -> str:
        return repr(self.value)
