# relative
from ..types.errors import SyftException


class SerdeError(SyftException): ...


class SerializationError(SerdeError):
    public_message = "Serialization failed."


class DeserializationError(SerdeError):
    public_message = "Deserialization failed."
