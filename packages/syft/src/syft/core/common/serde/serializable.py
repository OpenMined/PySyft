# stdlib
from typing import Any
from typing import Type

# third party
from google.protobuf.message import Message
from google.protobuf.reflection import GeneratedProtocolMessageType

# syft relative
from ....logger import traceback_and_raise
from ....util import random_name


def bind_protobuf(cls: Any) -> Any:
    protobuf_schema = cls.get_protobuf_schema()
    # If protobuf already has schema2type, means it's related to multiple types.
    # Set it's schema2type to None, becuase we can't take use of it anymore.
    if getattr(protobuf_schema, "schema2type", None):
        protobuf_schema.schema2type = None
    else:
        protobuf_schema.schema2type = cls

    return cls


class Serializable:
    """When we want a custom object to be serializable within the Syft ecosystem
    (as outline in the tutorial above), the first thing we need to do is have it
    subclass from this class. You must then do the following in order for the
    subclass to be properly implemented:

    - implement a protobuf file in the "PySyft/proto" folder for this custom class.
    - compile the protobuf file by running `bash scripts/build_proto`
    - find the generated python file in syft.proto
    - import the generated protobuf class into my custom class
    - implement get_protobuf_schema
    - implement <my class>._object2proto() method to serialize the object to protobuf
    - implement <my class>._proto2object() to deserialize the protobuf object

    At this point, your class should be ready to serialize and deserialize! Don't
    forget to add tests for your object!

    If you want to wrap an existing type (like a torch.tensor) to be used in our serialization
    ecosystem, you should consider wrapping it. Wrapping means that we NEVER use the wrapper
    further more into our ecosystem, we only need an easy interface to serialize wrappers.

    Eg:

    .. code-block:: python

        class WrapperInt(Serializable)
            def __init__(self, value: int):
               self.int_obj = value

            def _object2proto(self) -> WrapperIntPB:
               ...

            @staticmethod
            def _proto2object(proto) -> int:
               ...

            @staticmethod
            def get_protobuf_schema() -> GeneratedProtocolMessageType:
               ...

            @staticmethod
            def get_wrapped_type() -> type:
               return int

    You must implement the following in order for the subclass to be properly implemented to be
    seen as a wrapper:

    - everything presented in the first tutorial of this docstring.
    - implement get_wrapped_type to return the wrapped type.

    Note: A wrapper should NEVER be used in the codebase, these are only for serialization purposes
    on external objects.

    After doing all of the above steps, you can call something like sy.serialize(5) and be
    serialized using our messaging proto backbone.
    """

    @property
    def named(self) -> str:
        if hasattr(self, "name"):
            return self.name  # type: ignore
        else:
            return "UNNAMED"

    @property
    def class_name(self) -> str:
        return str(self.__class__.__name__)

    @property
    def icon(self) -> str:
        # as in cereal, get it!?
        return "🌾"

    @property
    def pprint(self) -> str:
        return f"{self.icon} {self.named} ({self.class_name})"

    @staticmethod
    def _proto2object(proto: Message) -> "Serializable":
        """This method converts a protobuf object into a subclass of Serializable

        This method must be implemented for all classes which subclassSerializable - namely
        all classes which can be serialized within the Syft ecosystem. It should convert the
        corresponding protobuf message for the subclass into an instance of the class. This
        allows all the logic which goes from protobuf message to other formats (JSON, binary, etc.)
        to be generic and simply inherited from the rest of this class.

        :param proto: the protobuf object to be converted into an instance of type(self)
        :param type: Message
        :return: an instance of type(self)
        :rtype: Serializable

        """
        traceback_and_raise(NotImplementedError)

    def _object2proto(self) -> Message:
        """This methods converts self into a protobuf object

        This method must be implemented by all subclasses so that generic high-level functions
        implemented here (such as serialize(, to_bytes=True), etc) know how to convert the object into
        a protobuf object before further converting it into the requested format.

        :return: a protobuf message
        :rtype: Message
        """

        traceback_and_raise(NotImplementedError)
        raise NotImplementedError

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        """Return the type of protobuf object which stores a class of this type

        As a part of serialization and deserialization, we need the ability to
        lookup the protobuf object type directly from the object type. This
        static method allows us to do this.

        Importantly, this method is also used to create the reverse lookup ability within
        the metaclass of Serializable. In the metaclass, it calls this method and then
        it takes whatever type is returned from this method and adds an attribute to it
        with the type of this class attached to it. See the MetaSerializable class for details.

        :return: the type of protobuf object which corresponds to this class.
        :rtype: GeneratedProtocolMessageType
        """

        traceback_and_raise(NotImplementedError)

    @staticmethod
    def get_wrapped_type() -> Type:
        """
        This static method returns the wrapped type, if the current class is
        a wrapper over an external object.

        :return: the wrapped type
        :rtype: type
        """
        traceback_and_raise(NotImplementedError)

    @staticmethod
    def random_name() -> str:
        return random_name()
