HANDLED_FUNCTIONS = {}


def query_implementation(tensor_type, func):
    name = func.__name__
    cache = HANDLED_FUNCTIONS[tensor_type]
    if name in cache:
        return HANDLED_FUNCTIONS[tensor_type][func.__name__]
    here = HANDLED_FUNCTIONS
    return None


def implements(tensor_type, np_function):
    def decorator(func):
        if tensor_type not in HANDLED_FUNCTIONS:
            HANDLED_FUNCTIONS[tensor_type] = {}

        HANDLED_FUNCTIONS[tensor_type][np_function.__name__] = func
        return func

    return decorator
