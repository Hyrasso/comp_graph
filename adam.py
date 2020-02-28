from src import *
from src.functions import Max
import random

class Adam:
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


x = Var("x")
y = Var("y")

w = Var("w")

def f(x):
    return x * 3

def relu(x):
    return Max(Const(0), x)

model = relu(x * w)

loss = (model - y) ** 2

opt = Adam(loss, w, 0.)
ll = 0
for i in range(100):
    for labels, targets in [(e/100, f(e/100)) for e in range(i, i+100)]:
        with loss.set_context({x:labels, y:targets}):
            opt.step()
            ll = loss({opt.parameters: opt.theta})
    print(i, opt.theta, ll)
