from abc import ABCMeta, abstractmethod

class BaseHook(object):
    __metaclass__ = ABCMeta
    """An abstract interface for deep learning framework hooks."""
    @abstractmethod
    def __init__(self):
        pass

    # TODO: decorate with abstractmethod after TorchHook is extended
    def __enter__(self):
        return self

    # TODO: decorate with abstractmethod after TorchHook is extended
    def __exit__(self):
        pass
