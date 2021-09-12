# stdlib
from typing import Any


def serializable(*args, **kwargs) -> Any:
    def serializable_decorator(cls: Any) -> Any:
        protobuf_schema = cls.get_protobuf_schema()
        # overloading a protobuf by adding multiple classes and we will check the
        # obj_type string later to dispatch to the correct one
        if hasattr(protobuf_schema, "schema2type"):
            if isinstance(protobuf_schema.schema2type, list):
                protobuf_schema.schema2type.append(cls)
            else:
                protobuf_schema.schema2type = [protobuf_schema.schema2type, cls]
        else:
            protobuf_schema.schema2type = cls
        return cls

    return serializable_decorator
