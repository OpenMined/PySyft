# stdlib
from typing import Callable
from typing import Optional


class ExperimentalFlags:
    def __init__(self) -> None:
        self._APACHE_ARROW_TENSOR_SERDE = True
        self._regenerate_numpy_serde: Optional[Callable] = None

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
        if self._regenerate_numpy_serde:
            self._regenerate_numpy_serde()


flags = ExperimentalFlags()
