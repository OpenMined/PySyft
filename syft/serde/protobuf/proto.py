"""
This file exists to translate python classes to and from Protobuf messages.
The reason for this is to have stable serialization protocol that can be used
not only by PySyft but also in other languages.

https://github.com/OpenMined/syft-proto (`syft_proto` module) is included as
a dependency in setup.py.
"""


def set_protobuf_id(field, id):
    if isinstance(id, str):
        field.id_str = id
    else:
        field.id_int = id


def get_protobuf_id(field):
    return getattr(field, field.WhichOneof("id"))
