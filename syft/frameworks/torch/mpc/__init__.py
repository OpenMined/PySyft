protocol_store = {}


def crypto_protocol(protocol_name):
    def decorator(f):
        name = f.__qualname__
        protocol_store[(name, protocol_name)] = f

        def method(self, *args, **kwargs):
            f = protocol_store[(name, self.protocol)]
            return f(self, *args, **kwargs)

        return method

    return decorator
