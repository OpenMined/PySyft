from abc import ABC, abstractmethod

class BaseHook(ABC):
    r""" A abstract interface for deep learning framework hooks."""
    @abstractmethod
    def __init__(self):
        pass

    # TODO: decorate with abstractmethod after TorchHook is extended
    def __enter__(self):
        raise NotImplementedError

    # TODO: decorate with abstractmethod after TorchHook is extended
    def __exit__(self):
        raise NotImplementedError
