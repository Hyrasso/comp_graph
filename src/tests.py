from operations import *
from function import *

x = Var("x")
y = Var("y")
a = Const(2)


z = x * a + x * y + x * x
z.set_context({x:1, y:2})
print(z.compute())
# Here differentiate should be a + 1 + y but only a + y is found, see differentiate method
dx = z.differentiate(x)
print(dx)
print(dx.compute())
# add reduce class
dzdy = z.differentiate(y)
print(z.differentiate(y))

