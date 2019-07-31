
class F:
    def derivate(self, var):
        if self == var:
            return Const(1)

        res = Const(0)
        # chain rule for multi variable
        # z = f + g
        # f can be const, variable, functions of anything
        # dz/dx = dz/df * df/dx + dz/dg * dg/dx 
        for i, arg in enumerate(self.args):
            # Replace with depth search 
            # currently fail to derivate if z = f(g(x))
            # recursive fonction, skip for now because of the var is defined
            if self == arg:
                continue
            dz_df = self.grad()[i]
            df_dvar = arg.derivate(var)
            res = res + dz_df * df_dvar
        return res
    
    def __add__(self, o):
        if not isinstance(o, F):
            o = Const(o)
        return Add(self, o)

    def __mul__(self, o):
        if not isinstance(o, F):
            o = Const(o)
        return Mul(self, o)

    def __rmul__(self, o):
        if not isinstance(o, F):
            o = Const(o)
        return Mul(o, self)

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, " ".join(map(repr, self.args)))


class Const(F):
    def __init__(self, a):
        self.args = tuple()
        self.a = a
    
    def eval(self):
        return self.a
    
    def grad(self):
        return (0,)
    
    def __repr__(self):
        return repr(self.a)
    

class Var(F):
    def __init__(self, name):
        self.name = name
        self.args = (self,)

    def eval(self):
        return self

    def grad(self):
        return (1,)
    
    def __repr__(self):
        return self.name
    
class Add(F):
    def __init__(self, a, b):
        self.args = (a, b)
        self.a = a
        self.b = b
    
    def eval(self):
        return self.a + self.b
    
    def grad(self):
        return (1, 1)
    
    def __repr__(self):
        return "(" + repr(self.a) + " + " + repr(self.b) + ")"

class Mul(F):
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.args = (a, b)
    
    def eval(self):
        return self.a * self.b
    
    def grad(self):
        return (self.b, self.a)

    def __repr__(self):
        return repr(self.a) + " * " + repr(self.b)
    
x = Var("x")
y = Var("y")
a = Const(2)


z = x * a + a + x + x * y

print z.eval()
# Here derivate should be a + 1 + y but only a + y is found, see derivate method
print z.derivate(x)
# add reduce class
print z.derivate(y)
