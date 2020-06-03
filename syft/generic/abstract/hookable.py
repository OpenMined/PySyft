from functools import wraps


def parse_return(raw_return):
    if not isinstance(raw_return, tuple):
        raw_return = (raw_return, [], {})
    return_val, *modified_args_kwargs = raw_return
    modified_args, *modified_kwargs = modified_args_kwargs
    modified_kwargs = modified_kwargs[0] if len(modified_kwargs) > 0 else {}
    return return_val, modified_args, modified_kwargs


def map_chain_call(obj, method_name, *args, **kwargs):
    """Calls each hook method in sequence and creates a list of the return values"""
    results = []
    current = obj
    args_ = args
    kwargs_ = kwargs
    while current is not None:
        method = getattr(current, method_name, None)
        if method:
            return_val, mod_args, mod_kwargs = parse_return(method(*args_, **kwargs_))
            results.append(return_val)
            args_ = mod_args if len(mod_args) == len(args_) else args_
            kwargs_ = mod_kwargs if len(mod_kwargs) == len(kwargs_) else kwargs_
        current = getattr(current, "child", None)
    return (results, args_, kwargs_)


def reduce_chain_call(obj, method_name, initial_val, *args, **kwargs):
    """Calls each hook method in sequence, passing return values from one to the next"""
    result = initial_val
    current = obj
    args_ = args
    kwargs_ = kwargs
    while current is not None:
        method = getattr(current, method_name, None)
        if method:
            return_val, mod_args, mod_kwargs = parse_return(method(result, *args_, **kwargs_))
            result = return_val
            args_ = mod_args if len(mod_args) == len(args_) else args_
            kwargs_ = mod_kwargs if len(mod_kwargs) == len(kwargs_) else kwargs_
        current = getattr(current, "child", None)
    return (result, args_, kwargs_)


def hookable(hookable_method):
    """Decorator which checks for corresponding hooks and calls them if they exist

    When this decorator is applied to a method, it checks for the existence of `_before_method()`
    and `_after_method()` hooks, and calls them before/after the correspdoning method if they do.

    This function should be used only as a decorator.
    """
    method_name = hookable_method.__name__

    @wraps(hookable_method)
    def hooked_method(self, *args, **kwargs):
        _, args_, kwargs_ = map_chain_call(self, f"_before_{method_name}", *args, **kwargs)
        return_val = hookable_method(self, *args_, **kwargs_)
        return_val, args_, kwargs_ = reduce_chain_call(
            self, f"_after_{method_name}", return_val, *args_, **kwargs_
        )
        return return_val

    return hooked_method
