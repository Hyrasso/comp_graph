from functools import wraps
from typing import Dict
import numbers

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

class EvaluationError(Exception):
    pass

class NodeClassError(Exception):
    pass

class Node:
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

    
    def __neg__(self):
        return Neg(self)
    
    def evaluate(self, context):
        return NotImplemented("Node cannot be evaluated needs to be subclassed")

class Variable(Node):
    def __init__(self, name: str=None):
        self.name = name

    def __repr__(self):
        return f"Variable({self.name})"
    
    def evaluate(self, context: Dict=None):
        if not context:
            raise EvaluationError(f"Can’t evaluate {self!r} without context")
        if self not in context["variables"]:
            simplification = context.get("simplification", False)
            if not simplification:
                raise EvaluationError(f"Can’t evaluate {self!r}, value not found")
            else:
                return self
        return context["variables"][self]

class Constant(Node):
    def __init__(self, value):
        self.value = value

    def evaluate(self, context):
        return self.value

    def __repr__(self):
        return f"Constant({self.value!r})"

class Operation(Node):
    """ Operation """
    def __init__(self, *args, **kwargs):
        self.args = args

    def evaluate(self, context):
        return NotImplemented("Operation cannot be evaluated needs to be subclassed")



class Add(Operation):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"Add({self.left!r}, {self.right!r})"

    def evaluate(self, context: Dict=None):
        """ Evalute and return the result of the operation """
        return self.left.evaluate(context) + self.right.evaluate(context)

class Sub(Operation):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"Sub({self.left!r}, {self.right!r})"

    def evaluate(self, context):
        return self.left.evaluate(context) - self.right.evaluate(context)

class Div(Operation):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"Div({self.left!r}, {self.right!r})"

    def evaluate(self, context):
        return self.left.evaluate(context) / self.right.evaluate(context)
        
class Mul(Operation):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"Mul({self.left!r}, {self.right!r})"

    def evaluate(self, context):
        return self.left.evaluate(context) * self.right.evaluate(context)

class Neg(Operation):
    def __init__(self, arg):
        self.arg = arg

    def __repr__(self):
        return f"Neg({self.arg!r})"

    def evaluate(self, context):
        return -self.arg.evaluate(context)

if __name__ == "__main__":
    x = Variable("x")
    y = Variable("y")
    a = Constant(2)

    z = x + y * a
    print(z)
    z = z / 2
    print(z)
    print(z.evaluate(variables={x: 2, y: 3}))