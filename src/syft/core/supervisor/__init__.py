def syft_supervisor(func):
    #@typecheck
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    #handle worker_stuff if it is worker
    ##enforce worker_policies

    #handle store_stuff if it is store
    ##enforce store_policies

    wrapper.__name__ = func.__name__
    wrapper.__qualname__ = func.__qualname__
    wrapper.__doc__ = func.__doc__

    return wrapper
