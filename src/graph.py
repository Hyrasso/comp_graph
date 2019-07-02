from typing import Dict
from functools import wraps
import operator

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
    def _add_methods(methods):
        "Add methods from class to Node objects"
        def decorator(cls):
            for method in methods:
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
    
    def get_nodes(self, filter_func=None):
        nodes = set(self)
        res = set()
        while nodes:
            current = nodes.pop()
            for node in filter(filter_func, current.args):
                if node not in res:
                    nodes.add(node)
            res.add(current)
        return nodes


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
