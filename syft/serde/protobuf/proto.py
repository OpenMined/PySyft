"""
This file exists to translate python classes to and from Protobuf messages.
The reason for this is to have stable serialization protocol that can be used
not only by PySyft but also in other languages.

https://github.com/OpenMined/syft-proto (`syft_proto` module) is included as
a dependency in setup.py.
"""
from google.protobuf.empty_pb2 import Empty


MAP_PYTHON_TO_PROTOBUF_CLASSES = {type(None): Empty}

MAP_PROTOBUF_TO_PYTHON_CLASSES = {}

for key, value in MAP_PYTHON_TO_PROTOBUF_CLASSES.items():
    MAP_PROTOBUF_TO_PYTHON_CLASSES[value] = key
