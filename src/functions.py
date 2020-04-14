from .node import Node
from src.variable import Const
import math

class Log(Node):
    def __init__(self, a: Node):
        self.a = a

    def compute(self):
        return math.log(self.a.compute())
    
    def grad(self):
        return 1 / self.a

class Dirac(Node):
    def compute(self):
        return 1 if self.args[0].compute() == 0 else 0
    
    def grad(self):
        return (Const(0),) # almost but actually not

class Step(Node):
    def compute(self):
        return 1 if self.args[0].compute() >= 0 else 0
    
    def grad(self):
        return (Dirac(self.args[0]),)


def Max(a, b):
    return Step(a - b) * a + Step(b - a) * b - Dirac(a - b) * a