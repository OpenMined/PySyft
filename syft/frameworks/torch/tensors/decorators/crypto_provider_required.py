from functools import wraps
from syft.exceptions import CryptoProviderNotFoundError


def crypto_provider_required(operation_name=""):
    """
    Decorator to verify if a crypto_provider was passed
    when a operation is applied to the shared tensor

    Args:
        operation_name: the name of the operation thats been applied.

    Example in a tensor file:
        ```
        @crypto_provider_required("add")
        def foo(crypto_provider = None):
            # ...

        return:
        raise CryptoProviderNotFoundError(
            "For this add operation, a crypto_provider must be passed.".format(operation_name)
        )
        ```

        See additive_sharing.py for more usage
    """

    def decorator(f):
        @wraps(f)
        def method(self, *args, **kwargs):
            if self.crypto_provider is None:
                raise CryptoProviderNotFoundError(
                    "For this {} operation, a crypto_provider must be passed.".format(
                        operation_name
                    )
                )
            else:
                return f(self, *args, **kwargs)

        return method

    return decorator
