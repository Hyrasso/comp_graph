from src import *
from src.operations import Add
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

batch_size = 10
theta = np.array([Var(f"t{i}") for i in range(input_size + 1)]).reshape((input_size + 1, 1))

xs = np.array(list([Const(1), Var(f"X{i} 1"), Var(f"X{i} 0")] for i in range(batch_size)))
ys = np.array(list([Const(1), Var(f"Y{i} 1"), Var(f"Y{i} 0")] for i in range(batch_size)))

# stochastic gradien descent with batch
def get_model(x):
    return x @ theta
# how to implement batch ?
# probably matmul is solution
# loss = sum(model({x:xi}) - target) ** 2) for xi in batch
# -> implement matmul, vector (1d tensor), and others, like ?sum
model = xs @ theta

loss = np.sum((model - ys) ** 2)

print(loss)

def f(x):
    return x @ np.array(((1,),  (0,)))
# mse = sum(loss for (x, y) in train) / len(train) 
lr = 0.001
theta_val = np.zeros((input_size + 1, 1))


theta_update = theta - loss.differentiate(theta) * lr
# print(m_update)
# class np_wrapper(np.ndarray):
#    def differentiate(var: F):
#       ...
#    def gradients(a: np.ndarray):
#       returns np.ndarray with same size as input with every element being
#       B[i,j,...] = self.differentiate(a[i,j,...])
#       B[a.shape + self.shape]
#     @contextmanager
#     def set_context(ctx):
#         ...
#     def update_value(key):
#         ...

for epoch in range(100):
    # x.value = ex
    # y.value = exy
    labels = np.random.random((100, 2)) * 10
    targets = f(labels)
    for i in range(0, len(labels), len(labels) // batch_size):
        ctx = {
            ctx_from_numpy(targets[i:i+batch_size]),
            ctx_from_numpy(targets[i:i+batch_size]),
            ctx_from_numpy(theta)
        }
        ctx.update({theta:theta_val})
        with loss.set_context(ctx):
            theta_val = theta_update.compute()
            if not i:print("loss", loss.compute())
print(m.value, b.value)
