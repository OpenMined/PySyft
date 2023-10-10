# stdlib
from enum import Enum
import os

# relative
from .util import str_to_bool


class ApacheArrowCompression(Enum):
    ZSTD = "zstd"
    LZ4 = "lz4"
    LZ4_RAW = "lz4_raw"
    BROTLI = "brotli"
    SNAPPY = "snappy"
    GZIP = "gzip"
    NONE = 0


class ExperimentalFlags:
    def __init__(self) -> None:
        self._APACHE_ARROW_TENSOR_SERDE = True
        self._APACHE_ARROW_COMPRESSION = ApacheArrowCompression.ZSTD
        self._CAN_REGISTER = str_to_bool(
            os.getenv(
                "ENABLE_SIGNUP",
                "False",
            )
        )

    @property
    def APACHE_ARROW_TENSOR_SERDE(self) -> bool:
        return self._APACHE_ARROW_TENSOR_SERDE

    @APACHE_ARROW_TENSOR_SERDE.getter
    def APACHE_ARROW_TENSOR_SERDE(self) -> bool:
        return self._APACHE_ARROW_TENSOR_SERDE

    @APACHE_ARROW_TENSOR_SERDE.setter
    def APACHE_ARROW_TENSOR_SERDE(self, value: bool) -> None:
        self._APACHE_ARROW_TENSOR_SERDE = value

    @property
    def APACHE_ARROW_COMPRESSION(self) -> ApacheArrowCompression:
        return self._APACHE_ARROW_COMPRESSION

    @APACHE_ARROW_COMPRESSION.setter
    def APACHE_ARROW_COMPRESSION(self, value: ApacheArrowCompression) -> None:
        self._APACHE_ARROW_COMPRESSION = value

    @property
    def USE_NEW_SERVICE(self) -> bool:
        return str_to_bool(os.getenv("USE_NEW_SERVICE", "False"))

    @property
    def CAN_REGISTER(self) -> bool:
        return self._CAN_REGISTER

    @CAN_REGISTER.setter
    def CAN_REGISTER(self, value: bool) -> None:
        self._CAN_REGISTER = value


flags = ExperimentalFlags()
