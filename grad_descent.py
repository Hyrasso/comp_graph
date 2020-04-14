from src import *
from src.tensor import *
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

batch_size = 64

theta = var(input_size).reshape((input_size, 1))
set_value(theta, np.random.random(theta.shape) * 2 - 1)
beta = var(1).reshape((1,))
set_value(beta, np.random.random((beta.shape)) * 2 - 1)

xs = var((batch_size, 2))
ys = var(batch_size).reshape((batch_size, 1))
print(xs)
# stochastic gradien descent with batch
def get_model(x):
    return x.dot(theta) + beta

model = get_model(xs)

loss = np.sum((model - ys) ** 2)

def f(x):
    coefs = np.array(((1,),  (2.5,)))
    return x @ coefs - 0.5

# mse = sum(loss for (x, y) in train) / len(train) 
lr = 0.001

theta_update = theta - gradients(loss, theta) * lr

for epoch in range(20):
    labels = np.random.random((batch_size * 10, input_size))
    targets = f(labels)
    for i in range(0, len(labels), len(labels) // batch_size):
        if labels[i:i+batch_size].shape[0] != batch_size:continue
        set_value(xs, labels[i:i+batch_size])
        set_value(ys, targets[i:i+batch_size])
        set_value(theta, compute(theta_update))
        if not i:print("loss", compute(loss))
print(compute(theta))
