from typing import Any, Tuple, Dict, List, Union

from .abstract import AbstractStorage


class MemoryStorage(AbstractStorage):

    def __init__(self):
        self.memory_dict = dict()

    def register(self, key: Any, value: Any) -> None:
        self.memory_dict[key] = value

    def get(self, **kwargs: Dict[Any, Any]) -> Union[ List[Any], None ]:
        raise NotImplementedError("get(**kwargs) isn't suported by MemoryStorage, Use get(*args) instead.")
    
    def get(self, *args: List[Any]) -> Union[Any, None]:
        return self.memory_dict.get(args[0], None)

    def __str__(self):
        return str(self.memory_dict)
