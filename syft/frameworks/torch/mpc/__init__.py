protocol_store = {}


def crypto_protocol(protocol_name):
    """
    Decorator to define a specific operation behaviour depending on the crypto
    protocol used

    Args:
        protocol_name: the name of the protocol. Currently supported:
            - snn: SecureNN
            - fss: Function Secret Sharing
            - falcon (WIP): Falcon

    Example in a tensor file:
        ```
        @crypto_protocol("snn")
        def foo(...):
            # SNN specific code

        @crypto_protocol("fss")
        def foo(...):
            # FSS specific code
        ```

        See additive_sharing.py for more usage
    """

    def decorator(f):
        name = f.__qualname__
        protocol_store[(name, protocol_name)] = f

        def method(self, *args, **kwargs):
            f = protocol_store[(name, self.protocol)]
            return f(self, *args, **kwargs)

        return method

    return decorator
