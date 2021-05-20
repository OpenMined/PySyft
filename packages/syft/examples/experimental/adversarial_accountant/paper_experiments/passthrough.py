import numpy as np

HANDLED_FUNCTIONS = {}

def inputs2child(*args, **kwargs):
    args = [x.child if isinstance(x, PassthroughTensor) else x for x in args]
    kwargs = {x.key : x.value.child if isinstance(x.value, PassthroughTensor) else x.value for x in kwargs.items()}
    return args,kwargs

class PassthroughTensor(np.lib.mixins.NDArrayOperatorsMixin):
    
    def __init__(self, child):
        self.child = child

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == '__call__':
            
            inputs, kwargs = inputs2child(*inputs, **kwargs)
            
            return  self.__class__(ufunc(*inputs, **kwargs))
        else:
            return NotImplemented
        
    def __array_function__(self, func, types, args, kwargs):
        args, kwargs = inputs2child(*args, **kwargs)
            
        # Note: this allows subclasses that don't override
        # __array_function__ to handle PassthroughTensor objects.
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented
        
        if func in HANDLED_FUNCTIONS:
            return self.__class__(HANDLED_FUNCTIONS[func](*args, **kwargs))
        else:
            return self.__class__(func(*args, **kwargs))
    
    def __repr__(self):
        return f"{self.__class__.__name__}(child={self.child})"

    
def implements(tensor_type, np_function):
        "Register an __array_function__ implementation for DiagonalArray objects."
        def decorator(func):
            if tensor_type not in HANDLED_FUNCTIONS:
                HANDLED_FUNCTIONS[tensor_type] = {}
                
            HANDLED_FUNCTIONS[tensor_type][np_function] = func
            return func
        return decorator