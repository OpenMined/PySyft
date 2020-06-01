from functools import wraps


def chain_call(obj, method_name, *args, **kwargs):
    results = []
    current = obj
    while current is not None:
        method = getattr(current, method_name, None)
        if method:
            results.append(method(*args, **kwargs))
        current = getattr(current, "child", None)
    return results


def hookable(hookable_method):
    """Decorator which checks for corresponding hooks and calls them if they exist

    When this decorator is applied to a method, it checks for the existence of `_before_method()`
    and `_after_method()` hooks, and calls them before/after the correspdoning method if they do.

    This function should be used only as a decorator.
    """
    method_name = hookable_method.__name__

    @wraps(hookable_method)
    def hooked_method(self, *args, **kwargs):
        chain_call(self, f"_before_{method_name}", *args, **kwargs)
        return_val = hookable_method(self, *args, **kwargs)
        chain_call(self, f"_after_{method_name}", *args, **kwargs)
        return return_val

    return hooked_method
