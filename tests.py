from src import *

x = Var("x")
y = Var("y")
a = Const(2)

z = x + y
print(z.grad())
print(z.differentiate(x))
print(z.differentiate(y))

# z = x * a - x * a / y + x ** x
# ctx = {x:1, y:2}
# z.set_context(ctx)
# print(z.compute())
# ctx[x] = 2
# print(z.compute())
# # Here differentiate should be a + 1 + y but only a + y is found, see differentiate method
# dx = z.differentiate(x)
# print(dx)
# print(dx.compute())
# # add reduce class
# dzdy = z.differentiate(y)
# print(dzdy)
# print(dzdy.compute())

