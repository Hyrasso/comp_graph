from src import *
from random import shuffle
# linear model
# y = mx + b
# vars -> m, b
# Loss -> sum((y - ÿ)**2) / N
# update state -> m = m - a * L * dL/dm

m = Var("m")
b = Var("b")
x = Var("x")
y = Var("y")

# stochastic gradien descent with batch size 1...
# result depends on shuffling
model = m * x + b
# how to implement batch ?
# probably matmul is solution
# or binding var for later?
# transmute var?
# loss = sum(model({x:xi}) - target) ** 2) for xi in batch
# if adding __call, do deepcopy of the graph and return with ctx values changed,
# kinda compiling the graph?
# loss is a function that takes a vector as entry
# -> implement matmul, vector (1d matrix), and others, like ?sum
loss = (model - y) ** 2

def f(x):
    return x * 1050 - 10
# mse = sum(loss for (x, y) in train) / len(train) 
lr = 0.001
m.value = 0.
b.value = 0.

training = [(x, f(x)) for x in range(50)]
shuffle(training)

m_update = m - loss.differentiate(m) * lr
b_update = b - loss.differentiate(b) * lr

for i, (ex, exy) in enumerate(training * 100):
    x.value = ex
    y.value = exy
    m.value = m_update.compute()
    b.value = b_update.compute()
    if i % 100 == 0:
        print(m, m.value)
        print(b, b.value)
        print("loss", loss.compute())
