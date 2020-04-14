from src.np_functions import gradients

class MinOptimizer:
    pass

class GradientDescent(MinOptimizer):
    def __init__(self, func, parameters, initial_value):
        self.f = func
        self.parameters = parameters
        self.theta = initial_value

        self.update_parameters = parameters - lr * func.differentiate(parameters) 
        self.lr = 0.001

    def step(self):
        with self.f.set_context({self.parameters: self.theta}):
            self.theta = self.update_parameters.compute()

class Adam(MinOptimizer):
    """https://arxiv.org/pdf/1412.6980.pdf"""
    def __init__(self, function, parameters, parameters_value):
        self.f = function
        self.parameters =  parameters
        self.theta = parameters_value
        self.m = 0
        self.v = 0

        self.a = 0.001
        self.b1 = 0.9
        self.b2 = 0.999
        self.e = 10e-8

        self.t = 0

    def step(self):
        self.t += 1
        f_grad = self.f.differentiate(self.parameters)
        with f_grad.set_context({self.parameters:self.theta}):
            g = f_grad.compute()
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g ** 2
        m_bc = self.m / (1 - self.b1 ** self.t)
        v_bc = self.v / (1 - self.b2 ** self.t)
        self.theta = self.theta - self.a * m_bc / (v_bc ** .5 + self.e)
        return self.theta
