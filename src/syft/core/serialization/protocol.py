from enum import Enum


class SerializationProtocol(Enum):
    ProtoBuffers: 0
    FlatBuffers: 1
    Custom: 2
