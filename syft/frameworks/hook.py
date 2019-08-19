from abc import ABC
from abc import abstractmethod

from syft.workers import BaseWorker


class BaseHook(ABC):
    def __init__(self, framework_module, local_worker: BaseWorker = None, is_client: bool = True):
        pass

    @classmethod
    @abstractmethod
    def create_wrapper(cls, child_to_wrap, *args, **kwargs):
        pass

    @classmethod
    @abstractmethod
    def create_shape(cls, shape_dims):
        pass
