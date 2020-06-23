from functools import wraps


def map_chain_call(obj, method_name, *args, **kwargs):
    """Calls each hook method in sequence and creates a list of the return values"""
    results = []
    current = obj
    while current is not None:
        method = getattr(current, method_name, None)
        if method:
            results.append(method(*args, **kwargs))
        current = getattr(current, "child", None)
    return results


def reduce_chain_call(obj, method_name, initial_val, *args, **kwargs):
    """Calls each hook method in sequence, passing return values from one to the next"""
    result = initial_val
    current = obj
    while current is not None:
        method = getattr(current, method_name, None)
        if method:
            result = method(result, *args, **kwargs)
        current = getattr(current, "child", None)
    return result


def hookable(hookable_method):
    """Decorator which checks for corresponding hooks and calls them if they exist

    When this decorator is applied to a method, it checks for the existence of `_before_method()`
    and `_after_method()` hooks, and calls them before/after the correspdoning method if they do.

    This function should be used only as a decorator.
    """
    method_name = hookable_method.__name__

    @wraps(hookable_method)
    def hooked_method(self, *args, **kwargs):
        map_chain_call(self, f"_before_{method_name}", *args, **kwargs)
        return_val = hookable_method(self, *args, **kwargs)
        return_val = reduce_chain_call(self, f"_after_{method_name}", return_val, *args, **kwargs)
        return return_val

    return hooked_method
