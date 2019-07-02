from typing import Dict
from functools import wraps
import operator

class NodeClassError(Exception):
    pass


# change to everything that is not a node becomes a Const
# No need to later add other types
# maybe deepcopy, would make more sense to not deepcopy
 
def args_to_const(func):
    """Decorator that wraps numbers.Number types from *args in Constant""" 
    def is_node(obj):
        return isinstance(obj, Node)

    @wraps(func)
    def wrapper(*args, **kwargs):
        args = (arg if is_node(arg) else Constant(arg) for arg in args)
        return func(*args, **kwargs)
    return wrapper

class Node:
    """Basic block of the computational graph"""
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, 'evaluate'):
            raise NodeClassError(f"Class '{cls.__name__}' doesn’t have an 'evaluate' mehtod")
        setattr(cls, 'evaluate', Node.wrap_evaluate(cls.evaluate))
    
    @staticmethod
    def _add_methods(methods):
        "Add methods from class to Node objects (only special methods starting with __)"
        def decorator(cls):
            for method in methods:
                if method.startswith("__"):
                    setattr(Node, method, getattr(cls, method))
            return cls
        return decorator

    @staticmethod
    def wrap_evaluate(method):
        @wraps(method)
        def wrapper(cls, *args, **kwargs):
            # Either context alone
            if len(args) > 1:
                raise TypeError(f"{method.__name__}() takes 1 positional argument but {len(args)} were given")
            # Or keywords argument alone
            if args and kwargs:
                raise TypeError(f"{method.__name__}() takes 1 positional argument (context) or keywords arguments, not both. {len(args)} positional arguments and {len(kwargs.keys())} keywords arguments were given")
            # Context
            if len(args) == 1:
                res =  method(cls, *args)
            else:
                # keywords
                res = method(cls, kwargs)
            return res
        return wrapper

    def evaluate(self, context):
        raise NotImplementedError

    # def grad(self):
    #     grad = {}
    #     for var in self.get_variables():
    #         grad[var] = self.derivate(var)
    #     return grad

class Constant(Node):
    """Constant, has a constant value set at instantiation"""
    def __init__(self, value):
        self.value = value

    def evaluate(self, context):
        return self.value

    def __repr__(self):
        return f"Constant({self.value!r})"
    
    def __str__(self):
        return f"{self.value!s}"

    def derivate(self, var):
        return 0

class EvaluationError(Exception):
    pass

# TODO: remove the simplification attribute from context
# Maybe add a simplify method later

class Variable(Node):
    """Variable, can be given a value at evaluation"""
    def __init__(self, name: str=None):
        self.name = name

    def __repr__(self):
        return f"Variable({self.name})"

    def __str__(self):
        return self.name

    def evaluate(self, context: Dict=None):
        if not context:
            raise EvaluationError(f"Can’t evaluate {self!r} without context")
        
        simplification = context.get("simplification", False)
        if "variables" not in context:
            if not simplification:
                raise EvaluationError(f"Can’t evaluate {self!r}, variables not found")
            else:
                return self
    
        if self not in context["variables"]:
            if not simplification:
                raise EvaluationError(f"Can’t evaluate {self!r}, value not found")
            else:
                return self
        return context["variables"][self]

    def derivate(self, var):
        if var is self:
            return 1
        else:
            return 0

# TODO: remove init and set args to Operation.args for every operation
#  maybe add accessor for first 2/3 args via xyz or abc
# It will make operation more general, no need to implement init
# option to set the number of excpected args 
# also possible to access the graph structure for (optimsation, …?¿)

class Operation(Node):
    """Base class for all operations"""
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
    
    def __repr__(self):
        args = map(lambda o:o.__repr__(), self.args)
        return f"{self.__class__.__name__}({' ,'.join(args)})"

    def evaluate(self, context):
        raise NotImplementedError

    def derivate(self, var):
        return NotImplementedError

@Node._add_methods(("__add__", "__radd__"))
class Add(Operation):
    def __init__(self, left, right):
        super().__init__(left, right)
        self.left = left
        self.right = right

    def evaluate(self, context: Dict=None):
        """ Evalute and return the result of the operation """
        return self.left.evaluate(context) + self.right.evaluate(context)
    
    def derivate(self, var):
        return self.left.derivate(var) + self.right.derivate(var)

    def __str__(self):
        return f"({self.left!s} + {self.right!s})"

    @args_to_const
    def __add__(self, other):
        return Add(self, other)
    
    @args_to_const
    def __radd__(self, other):
        return Add(other, self)

@Node._add_methods(("__sub__", "__rsub__"))
class Sub(Operation):
    def __init__(self, left, right):
        super().__init__(left, right)
        self.left = left
        self.right = right

    def __str__(self):
        return f"({self.left!s} - {self.right!s})"

    def evaluate(self, context):
        return self.left.evaluate(context) - self.right.evaluate(context)

    def derivate(self, var):
        return self.left.derivate(var) - self.right.derivate(var)

    @args_to_const
    def __sub__(self, other):
        return Sub(self, other)

    @args_to_const
    def __rsub__(self, other):
        return Sub(other, self)

@Node._add_methods(("__truediv__", "__rtruediv__"))
class Div(Operation):
    def __init__(self, left, right):
        super().__init__(left, right)
        self.left = left
        self.right = right

    def __str__(self):
        return f"({self.left!s} / {self.right!s})"

    def evaluate(self, context):
        return self.left.evaluate(context) / self.right.evaluate(context)
    
    def derivate(self, var):
        return Mul(self.left, Pow(self.right, Constant(-1))).derivate(var)

    @args_to_const
    def __truediv__(self, other):
        return Div(self, other)

    @args_to_const
    def __rtruediv__(self, other):
        return Div(other, self)

@Node._add_methods(("__mul__", "__rmul__"))
class Mul(Operation):
    def __init__(self, left, right):
        super().__init__(left, right)
        self.left = left
        self.right = right

    def __str__(self):
        return f"({self.left!s} * {self.right!s})"

    def evaluate(self, context):
        return self.left.evaluate(context) * self.right.evaluate(context)
    
    def derivate(self, var):
        if isinstance(self.left, Variable) and isinstance(self.right, Variable) and var in (self.left, self.right):
            return self.left if var == self.right else self.right
        if isinstance(self.left, Constant) and isinstance(self.right, Constant):
            return self.evaluate(simplification=True)
        if isinstance(self.left, (Operation, Variable)) and isinstance(self.right, (Operation, Variable)):
            f = self.left
            fp = f.derivate(var) 
            g = self.right
            gp = g.derivate(var)
            return fp * g.evaluate(simplification=True) + f.evaluate(simplification=True) * gp
        
        if isinstance(self.left, (Operation, Variable)) and isinstance(self.right, Constant):
            return self.left.derivate(var) * self.right.evaluate(simplification=True)
        
        if isinstance(self.left, Constant) and isinstance(self.right, (Operation, Variable)):
            return self.left.evaluate(simplification=True) * self.right.derivate(var)

        raise NotImplementedError

    @args_to_const
    def __mul__(self, other):
        return Mul(self, other)
    
    @args_to_const
    def __rmul__(self, other):
        return Mul(other, self)

@Node._add_methods(("__pow__", "__rpow__"))
class Pow(Operation):
    def __init__(self, left, right):
        super().__init__(left, right)
        self.left = left
        self.right = right

    def __str__(self):
        return f"({self.left!s} ** {self.right!s})"

    def evaluate(self, context):
        return self.left.evaluate(context) ** self.right.evaluate(context)
    
    def derivate(self, var):
        if isinstance(self.left, Operation) and isinstance(self.right, Constant):
            n = self.right.evaluate(simplification=True)
            return n * Pow(self.left.evaluate(simplification=True), Constant(n - 1)) * self.right.derivate(var)

        if isinstance(self.left, Variable) and isinstance(self.right, Constant):
            n = self.right.evaluate(simplification=True)
            return n * Pow(self.left.evaluate(simplification=True), Constant(n - 1))

        raise NotImplementedError
    
    @args_to_const
    def __pow__(self, other):
        return Pow(self, other)

    @args_to_const
    def __rpow__(self, other):
        return Pow(other, self)

@Node._add_methods(("__neg__",))
class Neg(Operation):
    def evaluate(self, context):
        return - self.args[0].evaluate(context)

    def derivate(self, var):
        return - self.args[0].derivate(var)

    def __neg__(self):
        return Neg(self)

if __name__ == "__main__":
    x = Variable("x")
    y = Variable("y")
    z = Variable("z")
    a = Constant(2)

    res = x ** -2 + y * z ** a / z + y
    print(res)
    print(res.derivate(x))
    print(res.evaluate(variables={x: 2, y: 3, z: 5}))
