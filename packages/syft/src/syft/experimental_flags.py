class ExperimentalFlags:
    def __init__(self) -> None:
        self._APACHE_ARROW_TENSOR_SERDE = True

    @property
    def APACHE_ARROW_TENSOR_SERDE(self) -> bool:
        return self._APACHE_ARROW_TENSOR_SERDE

    @APACHE_ARROW_TENSOR_SERDE.getter
    def APACHE_ARROW_TENSOR_SERDE(self) -> bool:
        return self._APACHE_ARROW_TENSOR_SERDE

    @APACHE_ARROW_TENSOR_SERDE.setter
    def APACHE_ARROW_TENSOR_SERDE(self, value: bool) -> None:
        self._APACHE_ARROW_TENSOR_SERDE = value


flags = ExperimentalFlags()
