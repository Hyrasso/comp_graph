from src import *
x = Var("x")
y = Var("y")
a = Const(2)

z = x ** 2 + y * a
print(z)
# ((x ^ 2) + (y * 2))
z = z / 2
print(repr(z))
# Div(Add(Pow(x, 2), Mul(y, 2)), 2)
x.value = 2
y.value = 3
# 5.0
print(z())

dzdx = z.differentiate(x)
print(dzdx)
# (((1 / 2) * ((1 * (((2 * (x ^ (2 - 1))) * 1) + (((x ^ 2) * Log(2)) * 0))) + (1 * ((2 * 0) + (y * 0))))) + ((-((x ^ 2) + (y * 2))) / (2 ^ 2)) * 0))
print(dzdx.compute())
# 2.0
with z.set_context({x:2, y:1}):
    print(z.differentiate(y).compute())
# 1.0