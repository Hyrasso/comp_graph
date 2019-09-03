from src import *
from src.operations import Add
from random import shuffle
import numpy as np
from functools import reduce

def var_tensor(shape, base_name="A"):
    if len(shape) == 1:
        return [Var(f"{base_name}{i}") for i in range(shape[0])]
    else:
        return [var_tensor(shape[1:], base_name=base_name + " i") for i in range(shape[1])]

def iter_sum(vec):
    return reduce(Add, (e for e in vec))

# def scal_tensor_mul(scal, tensor):
#     for e in tensor:
#         e *= scal

# linear model
# y = mx + b
# vars -> m, b
# Loss -> sum((y - ÿ)**2) / N
# update state -> m = m - a * L * dL/dm
batch_size = 10
m = Var("m")
b = Var("b")
xs = var_tensor((batch_size,), "X")
ys = var_tensor((batch_size,), "Y")

# stochastic gradien descent with batch
def get_model(x):
    return m * x + b
# how to implement batch ?
# probably matmul is solution
# loss = sum(model({x:xi}) - target) ** 2) for xi in batch
# -> implement matmul, vector (1d tensor), and others, like ?sum

loss = iter_sum((get_model(x) - y) ** 2 for x, y in zip(xs, ys))

print(loss)

def f(x):
    return x * 1050 - 10
# mse = sum(loss for (x, y) in train) / len(train) 
lr = 0.001
m.value = 0.
b.value = 0.


m_update = m - loss.differentiate(m) * lr
b_update = b - loss.differentiate(b) * lr
# print(m_update)

for epoch in range(50):
    # x.value = ex
    # y.value = exy
    labels = np.random.random(100) * 10
    targets = f(labels)
    for i in range(len(labels) // batch_size):
        ctx = dict(zip(xs, labels[i*batch_size:(i+1)*batch_size]))
        ctx.update(dict(zip(ys, targets[i*batch_size:(i+1)*batch_size])))
        loss.set_context(ctx)
        m.value = m_update.compute()
        b.value = b_update.compute()
    print(m, m.value)
    print(b, b.value)
    print("loss", loss.compute())
