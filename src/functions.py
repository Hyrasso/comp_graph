import math

from graph import Operation
from operations import Pow

class Exp(Operation):
    e = math.e

    def evaluate(self, context):
        return Pow(math.e, self.args[0])
    
    def derivate(self, var):
        return self.args[0].derivate(var) * self.evaluate()