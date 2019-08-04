
# linear model
# y = mx + b
# vars -> m, b
# Loss -> sum((y - ÿ)**2) / N
# update state -> m = m - a * L * dL/dm

m = Var()
b = Var()
x = Var()
target = Var()

y = m * x + b
loss = (y - target) ** 2

# mse = sum(loss for (x, y) in train) / len(train) 
lr = 0.1
m = m - loss.dervative(m) * lr 
b = b - loss.dervative(b) * lr
