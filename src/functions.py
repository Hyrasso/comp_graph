from .F import F, Const
import math

class Log(F):
    def __init__(self, a: F, **kwargs):
        super().__init__(a, **kwargs)
        self.a = a

    def compute(self):
        return math.log(self.a.compute())
    
    def grad(self):
        return 1 / self.a
