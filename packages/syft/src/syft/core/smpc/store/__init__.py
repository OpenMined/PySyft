"""Package where needed crypto material is stored."""

# stdlib
# stdlib
from typing import Any
from typing import Callable

# relative
from .crypto_primitive_provider import CryptoPrimitiveProvider
from .crypto_store import CryptoStore


def register_primitive_generator(name: str) -> Callable[..., Any]:
    """Decorator to register a crypto primitive provider.

    Args:
        name (str): Name of the primitive.

    Returns:
        Callable[..., Any]: returns the triple generator function.
    """

    def register_generator(func_generator: Callable[..., Any]) -> Callable[..., Any]:
        if name in CryptoPrimitiveProvider._func_providers:
            raise ValueError(f"Provider {name} already in _func_providers")
        CryptoPrimitiveProvider._func_providers[name] = func_generator
        return func_generator

    return register_generator


def register_primitive_store_add(name: str) -> Callable[..., Any]:
    """Decorator to add primitives to the store.

    Args:
        name (str): Name of the primitive.

    Returns:
        Callable[..., Any]: returns the crypto store add function.
    """

    def register_add(func_add: Callable[..., Any]) -> Callable[..., Any]:
        if name in CryptoStore._func_add_store:
            raise ValueError(f"Crypto Store 'adder' {name} already in _func_add_store")
        CryptoStore._func_add_store[name] = func_add
        return func_add

    return register_add


def register_primitive_store_get(name: str) -> Callable[..., Any]:
    """Decorator to retrieve primitives from the store.

    Args:
        name (str): Name of the primitive.

    Returns:
        Callable[..., Any]: returns the crypto store get function.
    """

    def register_get(func_get: Callable[..., Any]) -> Callable[..., Any]:
        if name in CryptoStore._func_get_store:
            raise ValueError(f"Crypto Store 'getter' {name} already in _func_get_store")
        CryptoStore._func_get_store[name] = func_get
        return func_get

    return register_get


__all__ = ["CryptoStore", "CryptoPrimitiveProvider"]
