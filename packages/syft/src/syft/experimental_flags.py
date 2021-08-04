
class ExperimentalFlags:
    def __init__(self) -> None:
        self._APACHE_ARROW_TENSOR_SERDE = True
        self._APACHE_ARROW_FLIGHT_CHANNEL = True
        self._FLIGHT_CHANNEL_PORT = 8999

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
    def APACHE_ARROW_FLIGHT_CHANNEL(self) -> bool:
        return self._APACHE_ARROW_FLIGHT_CHANNEL

    @APACHE_ARROW_FLIGHT_CHANNEL.getter
    def APACHE_ARROW_FLIGHT_CHANNEL(self) -> bool:
        return self._APACHE_ARROW_FLIGHT_CHANNEL

    @APACHE_ARROW_FLIGHT_CHANNEL.setter
    def APACHE_ARROW_FLIGHT_CHANNEL(self, value: bool) -> None:
        self._APACHE_ARROW_FLIGHT_CHANNEL = value

    @property
    def FLIGHT_CHANNEL_PORT(self) -> int:
        return self._FLIGHT_CHANNEL_PORT

    @FLIGHT_CHANNEL_PORT.getter
    def FLIGHT_CHANNEL_PORT(self) -> int:
        return self._FLIGHT_CHANNEL_PORT

    @FLIGHT_CHANNEL_PORT.setter
    def FLIGHT_CHANNEL_PORT(self, value: int) -> None:
        self._FLIGHT_CHANNEL_PORT = value

flags = ExperimentalFlags()