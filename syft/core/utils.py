def map_tuple(service, args, func):
    if service:
        return tuple(func(service, x) for x in args)
    else:
        return tuple(func(x) for x in args)


def map_dict(service, kwargs, func):
    if service:
        return {key: func(service, val) for key, val in kwargs.items()}
    else:
        return {key: func(val) for key, val in kwargs.items()}