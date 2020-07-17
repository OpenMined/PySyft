def supervisor(func):
    #@typecheck
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    #handle worker_stuff
    ##enforce worker_policies

    #handle store_stuff
    ##enforce store_policies
    return wrapper