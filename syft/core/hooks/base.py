from abc import ABCMeta, abstractmethod

class BaseHook(object):
    __metaclass__ = ABCMeta
    """An abstract interface for deep learning framework hooks."""
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __enter__(self):
        return self

    @abstractmethod
    def __exit__(self):
        pass
