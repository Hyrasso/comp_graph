from functools import wraps

from graph import Node, Constant

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
