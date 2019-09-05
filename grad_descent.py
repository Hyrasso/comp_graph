from src import *
from src.np_functions import *
from random import shuffle
import numpy as np
from functools import reduce

# linear model
# y = mx + b
# vec notation
# Y = XM
# avec 1 dans la premiére colonne de X pour le biais
# vars -> m, b
# Loss -> sum((y - ÿ)**2) / N
# update state -> m = m - a * L * dL/dm
input_size = 2
# function to model R2 -> R

batch_size = 20

theta = np.array([Var(f"t{i}") for i in range(input_size + 1)]).reshape((input_size + 1, 1))

xs = np.array(list([Const(1), Var(f"X{i} 1"), Var(f"X{i} 0")] for i in range(batch_size)))
ys = np.array([Var(f"Y{i}") for i in range(batch_size)]).reshape((batch_size, 1))

# stochastic gradien descent with batch
def get_model(x):
    return x @ theta
# how to implement batch ?
# probably matmul is solution
# loss = sum(model({x:xi}) - target) ** 2) for xi in batch
# -> implement matmul, vector (1d tensor), and others, like ?sum
model = xs @ theta

loss = np.sum((model - ys) ** 2)

def f(x):
    coefs = np.array(((1,),  (2.5,)))
    return x @ coefs - 0.5

# mse = sum(loss for (x, y) in train) / len(train) 
lr = 0.001

theta_update = theta - gradients(loss, theta) * lr
theta_val = make_value(theta)

for epoch in range(100):
    labels = np.random.random((batch_size * 10, input_size))
    targets = f(labels)
    for i in range(0, len(labels), len(labels) // batch_size):
        if labels[i:i+batch_size].shape[0] != batch_size:continue
        ctx = {
            **ctx_from_array(xs[:,1:], labels[i:i+batch_size]),
            **ctx_from_array(ys, targets[i:i+batch_size]),
            **ctx_from_array(theta, theta_val)
        }
        with loss.set_context(ctx):
            theta_val = compute(theta_update)
            if not i:print("loss", loss.compute())
print(theta_val)
