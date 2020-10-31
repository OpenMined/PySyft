# stdlib
from typing import Any
from typing import Dict
from typing import Tuple
from typing import Type
from typing import Union

# third party
from google.protobuf.message import Message
from google.protobuf.reflection import GeneratedProtocolMessageType
from loguru import logger

# Fixes python3.6
# however API changed between versions so typing_extensions smooths this over:
# https://cirq.readthedocs.io/en/stable/_modules/typing_extensions.html
from typing_extensions import GenericMeta as GenericM  # type: ignore

# syft relative
from ....decorators import syft_decorator
from ....proto.util.data_message_pb2 import DataMessage
from ....util import get_fully_qualified_name
from ....util import random_name


# GenericMeta Fixes python 3.6
# After python 3.7+ there is no GenericMeta only Generic and this becomes "type"
# python 3.6 print(typing_extensions.GenericMeta) = <class 'typing.GenericMeta'>
# python 3.7 print(typing_extensions.GenericMeta) = <class 'type'>
class MetaSerializable(GenericM):

    """When we go to deserialize a JSON protobuf object, the JSON protobuf
    wrapper will return a python protobuf object corresponding to a subclass
    of Serializable. However, in order to be able to take the next step, we need
    an instance of the Serializable subclass. In order to create this instance,
    we cache/monkeypatch it onto the protobuf class it corresponds to.

    Since this could be a dangerous thing to do (because developers of new objects
    in Syft could forget to add the schema2type attribute) we do it automatically
    for all subclasses of Serializable via this metaclass. This way, nobody has
    to worry about remembering to implement this flag."""

    def __new__(
        cls: Type, name: str, bases: Tuple[Type, ...], dct: Dict[str, Any]
    ) -> "MetaSerializable":
        x = super().__new__(cls, name, bases, dct)
        try:
            protobuf_schema = dct["get_protobuf_schema"].__get__("")()
            protobuf_schema.schema2type = x
        except (KeyError, NotImplementedError):
            ""
        return x


class Serializable(metaclass=MetaSerializable):
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

    class WrapperInt(Serializable):
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
        raise NotImplementedError

    @staticmethod
    @syft_decorator(typechecking=True)
    def _object2proto() -> Message:
        """This methods converts self into a protobuf object

        This method must be implemented by all subclasses so that generic high-level functions
        implemented here (such as .binary(), etc) know how to convert the object into
        a protobuf object before further converting it into the requested format.

        :return: a protobuf message
        :rtype: Message
        """

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

        raise NotImplementedError

    @staticmethod
    def get_wrapped_type() -> Type:
        """
        This static method returns the wrapped type, if the current class is
        a wrapper over an external object.

        :return: the wrapped type
        :rtype: type
        """
        raise NotImplementedError

    @syft_decorator(typechecking=True)
    def to_proto(self) -> Message:
        """A convenience method to convert any subclass of Serializable into a protobuf object.

        :return: a protobuf message
        :rtype: Message
        """
        return self.serialize(to_proto=True)

    @syft_decorator(typechecking=True)
    def proto(self) -> Message:
        """A convenience method to convert any subclass of Serializable into a protobuf object.

        :return: a protobuf message
        :rtype: Message
        """
        return self.serialize(to_proto=True)

    @syft_decorator(typechecking=True)
    def to_bytes(self) -> bytes:
        """A convenience method to convert any subclass of Serializable into a binary object.

        :return: a binary string
        :rtype: bytes
        """
        return self.serialize(to_bytes=True)

    @syft_decorator(typechecking=True)
    def binary(self) -> bytes:
        """A convenience method to convert any subclass of Serializable into a binary object.

        :return: a binary string
        :rtype: bytes
        """
        return self.serialize(to_bytes=True)

    @syft_decorator(typechecking=True)
    def serialize(
        self,
        to_proto: bool = True,
        to_bytes: bool = False,
    ) -> Union[str, bytes, Message]:
        """Serialize the object according to the parameters.

        This is the primary serialization method, which processes the above
        flags in a particular order. In general, it is not expected that people
        will set multiple to_<type> flags to True at the same time. We don't
        currently have logic which prevents this, because this may affect
        runtime performance, but if several flags are True, then we will simply
        take return the type of latest supported flag from the following list:

            - proto
            - json
            - binary
            - hex

        TODO: we could also add "dict" to this list but it's not clear if it would be used.

        :param to_proto: set this flag to TRUE if you want to return a protobuf object
        :type to_proto: bool
        :param to_bytes: set this flag to TRUE if you want to return a binary object
        :type to_bytes: bool
        :return: a serialized form of the object on which serialize() is called.
        :rtype: Union[str, bytes, Message]

        """

        if to_bytes:
            logger.debug(f"Serializing {type(self)}")
            # indent=None means no white space or \n in the serialized version
            # this is compatible with json.dumps(x, indent=None)
            blob = self._object2proto().SerializeToString()
            blob = DataMessage(
                obj_type=get_fully_qualified_name(obj=self), content=blob
            )
            return blob.SerializeToString()

        elif to_proto:
            return type(self)._object2proto(self)
        else:
            raise Exception(
                """You must specify at least one deserialization format using
                            one of the arguments of the serialize() method such as:
                            to_proto, to_bytes."""
            )

    @staticmethod
    def random_name() -> str:
        return random_name()
