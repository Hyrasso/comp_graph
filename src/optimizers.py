from src.np_functions import gradients

class MinOptimizer:
    pass

class GradientDescent(MinOptimizer):
    def __init__(self, func, parameters, initial_value):
        self.f = func
        self.parameters = parameters
        self.theta = initial_value

        self.update_parameters = parameters - lr * gradients(func, parameters) 
        self.lr = 0.001

    def step(self):
        with self.f.set_context({self.parameters: self.theta}):
            self.theta = self.update_parameters.compute()
