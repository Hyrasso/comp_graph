from typing import Dict
from functools import wraps
import numbers

class NodeClassError(Exception):
    pass

# Weird cyclic dependence
# NumericOperators depends on all operations and Const AND is baseclass for all nodes
# Fixes : rethink architecture, monkey patch

def numbers_to_const(func):
    """Decorator that wraps numbers.Number types from *args in Constant""" 
    def is_number(obj):
        return isinstance(obj, numbers.Number)

    @wraps(func)
    def wrapper(*args, **kwargs):
        non_numbers = list(filter(lambda o: not is_number(o), args))
        numbers = list(map(Constant, filter(is_number, args)))
        return func(*non_numbers, *numbers, **kwargs)
    return wrapper

class NumericOperators:
    """Class/interface implementing all the methods to emulate numeric types of python"""
    @numbers_to_const
    def __add__(self, right):
        return Add(self, right)

    @numbers_to_const
    def __sub__(self, right):
        return Sub(self, right)

    @numbers_to_const
    def __mul__(self, right):
        return Mul(self, right)

    @numbers_to_const
    def __truediv__(self, right):
        return Div(self, right)

    @numbers_to_const
    def __pow__(self, left):
        return Pow(self, left)

    @numbers_to_const
    def __radd__(self, left):
        return Add(left, self)

    @numbers_to_const
    def __rsub__(self, left):
        return Sub(left, self)

    @numbers_to_const
    def __rmul__(self, left):
        return Mul(left, self)

    @numbers_to_const
    def __rtruediv__(self, left):
        return Div(left, self)

    @numbers_to_const
    def __rpow__(self, left):
        return Pow(left, self)

    def __neg__(self):
        return Neg(self)

class Node(NumericOperators):
    """Basic block of the computational graph"""
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, 'evaluate'):
            raise NodeClassError(f"Class '{cls.__name__}' doesn’t have an 'evaluate' mehtod")
        setattr(cls, 'evaluate', Node.wrap_evaluate(cls.evaluate))
    
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


class Constant(Node):
    """Constant, has a constant value set at instantiation"""
    def __init__(self, value):
        self.value = value

    def evaluate(self, context):
        return self.value

    def __repr__(self):
        return f"Constant({self.value!r})"

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

class Operation(Node):
    """Base class for all operations"""
    def __init__(self, *args, **kwargs):
        self.args = args

    def evaluate(self, context):
        raise NotImplementedError

class Add(Operation):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"Add({self.left!r}, {self.right!r})"

    def evaluate(self, context: Dict=None):
        """ Evalute and return the result of the operation """
        return self.left.evaluate(context) + self.right.evaluate(context)
    
    def derivate(self, var):
        return self.left.derivate(var) + self.right.derivate(var)

class Sub(Operation):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"Sub({self.left!r}, {self.right!r})"

    def evaluate(self, context):
        return self.left.evaluate(context) - self.right.evaluate(context)

    def derivate(self, var):
        return self.left.derivate(var) - self.right.derivate(var)

class Div(Operation):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"Div({self.left!r}, {self.right!r})"

    def evaluate(self, context):
        return self.left.evaluate(context) / self.right.evaluate(context)
    
    def derivate(self, var):
        return Mul(self.left, Pow(self.right, Constant(-1))).derivate(var)

class Mul(Operation):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"Mul({self.left!r}, {self.right!r})"

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
# TODO: remove init and set args to Operation.args for every operation
#  maybe add accessor for first 2/3 args via xyz or abc
# It will make operation more general, no need to implement init
# also possible to access the graph structure for (optimsation, …?¿)

class Pow(Operation):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"Pow({self.left!r}, {self.right!r})"

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

class Neg(Operation):
    def __init__(self, arg):
        self.arg = arg

    def __repr__(self):
        return f"Neg({self.arg!r})"

    def evaluate(self, context):
        return -self.arg.evaluate(context)
    
    def derivate(self, var):
        return - self.arg.derivate(var)


if __name__ == "__main__":
    x = Variable("x")
    y = Variable("y")
    z = Variable("z")
    a = Constant(2)

    res = x ** -2 + y * z ** a / z + y
    print(res)
    print(res.derivate(x))
    print(res.evaluate(variables={x: 2, y: 3, z: 5}))
