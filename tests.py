from src import *

x = Var("x")
y = Var("y")
a = Const(2)

z = x ** 2 + y * a
print(repr(z))
# Add(Variable(x), Mul(Variable(y), Constant(2)))
z = z / 2
print(z)
# Div(Add(Variable(x), Mul(Variable(y), Constant(2))), Constant(2))
print(z.compute({x: 2, y: 3}))
# 4.0

dzdx = z.differentiate(x)
print(dzdx)
print(dzdx.compute())
print(z.differentiate(y).compute({x:2, y:1}))

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

