# stdlib
import importlib
from types import ModuleType
from typing import Set


class ExperimentalFlags:
    def __init__(self) -> None:
        self._APACHE_ARROW_TENSOR_SERDE = True
        self.apache_arrow_modules: Set[ModuleType] = set()

    @property
    def APACHE_ARROW_TENSOR_SERDE(self) -> bool:
        return self._APACHE_ARROW_TENSOR_SERDE

    @APACHE_ARROW_TENSOR_SERDE.getter
    def APACHE_ARROW_TENSOR_SERDE(self) -> bool:
        return self._APACHE_ARROW_TENSOR_SERDE

    @APACHE_ARROW_TENSOR_SERDE.setter
    def APACHE_ARROW_TENSOR_SERDE(self, value: bool) -> None:
        if self._APACHE_ARROW_TENSOR_SERDE == value:
            return

        self._APACHE_ARROW_TENSOR_SERDE = value
        for module in self.apache_arrow_modules:
            importlib.reload(module)


flags = ExperimentalFlags()
