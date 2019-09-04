from .F import F, Const
import math

class Log(F):
    def __init__(self, a: F):
        self.a = a

    def compute(self):
        return math.log(self.a.compute())
    
    def grad(self):
        return 1 / self.a

class Dirac(F):
    def compute(self):
        return 1 if self._args[0].compute() == 0 else 0
    
    def grad(self):
        return (Const(0),) # almost but actually not

class Step(F):
    def compute(self):
        return 1 if self._args[0].compute() >= 0 else 0
    
    def grad(self):
        return (Dirac(self._args[0]),)


def Max(a, b):
    return Step(a - b) * a + Step(b - a) * b - Dirac(a - b) * a