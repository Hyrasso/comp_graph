from typing import Any, Tuple, Optional
from .F import F, Const

class Var(F):
    def __init__(self, name: str = "Var"):
        self.name = name

    def compute(self) -> F:
        """ Returns value if it exits in F.context, otherwise returns itself """
        return F.context.get(self, self)

    # Never called because differentiate is implemented directly 
    def grad(self):
        return (Const(1),)

    def __repr__(self):
        return self.name
    
    def differentiate(self, var):
        return Const(1) if self == var else Const(0)

    # overload F.__eq__, var is only equal to itself
    def __eq__(self, value):
        return self is value
    
    # if eq is defined, hash must be too?
    def __hash__(self):
        return super().__hash__()
