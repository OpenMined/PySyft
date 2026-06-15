from pydantic import BaseModel
from typing import Dict, List, Callable


class BaseModelCallbackMixin(BaseModel):
    callbacks: Dict[str, List[Callable]] = {}

    def add_callback(self, on: str, callback: Callable):
        if on not in self.callbacks:
            self.callbacks[on] = []
        self.callbacks[on].append(callback)
