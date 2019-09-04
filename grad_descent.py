from src import *
from src.operations import Add
from random import shuffle
import numpy as np
from functools import reduce

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
xs = np.array(list(Var(f"X{i}") for i in batch_size))
ys = np.array(list(Var(f"Y{i}") for i in batch_size))

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
    return x * 10.50 - 1.2
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
        ctx.update({m:m.value, b:b.value})
        ctx.update(dict(zip(ys, targets[i*batch_size:(i+1)*batch_size])))
        with loss.set_context(ctx):
            m.value = m_update.compute()
            b.value = b_update.compute()
    print(m, m.value)
    print(b, b.value)
    with loss.set_context(ctx):
        print("loss", loss.compute())
