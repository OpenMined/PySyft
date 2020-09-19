# external class/method imports (sorted by length)
from google.protobuf.message import Message
from google.protobuf import json_format

# syft import
from syft.decorators.syft_decorator_impl import syft_decorator


class Serializable(object):
    """When we want a custom object to be serializable within the Syft ecosystem
    (as outline in the tutorial above), the first thing we need to do is have it
    subclass from this class. You must then do the following in order for the
    subclass to be properly implemented:

    - implement a protobuf file in the "PySyft/proto" folder for this custom class.
    - compile the protobuf file by running `bash scripts/build_proto`
    - find the generated python file in syft.proto
    - import the generated protobuf class into my custom class
    - set <my class>.protobuf_type = <generated protobuf python class>
    - implement <my class>._object2proto() method to serialize the object to protobuf
    - implement <my class>._proto2object() to deserialize the protobuf object

    At this point, your class should be ready to serialize and deserialize! Don't
    forget to add tests for your object!
    """

    def __init__(self, as_wrapper: bool):
        """In the initializer for this class, we check that the protobuf_type was
        properly set and save the as_wrapper parameter.

        :param as_wrapper: if set to true, it means that this outer object is merely
            serving to wrap an object which we couldn't subclass from Serializable
            because it is not a native Syft object (such as a torch.Tensor).
        :type as_warpper: bool
        """

        # check to make sure protobuf_type has been set on the class
        assert self.protobuf_type is not None

        # set the as_wrapper flag
        self.as_wrapper = as_wrapper

    @staticmethod
    def _proto2object(proto: Message) -> "Serializable":
        """This method converts a protobuf object into a subclass of Serializable

        This method must be implemented for all classes which subclass Serializable - namely
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

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> Message:
        """This methods converts self into a protobuf object

        This method must be implemented by all subclasses so that generic high-level functions
        implemented here (such as .json(), .binary(), etc) know how to convert the object into
        a protobuf object before further converting it into the requested format.

        :return: a protobuf message
        :rtype: Message
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
    def to_json(self) -> str:
        """A convenience method to convert any subclass of Serializable into a JSON object.

        :return: a JSON string
        :rtype: str
        """
        return self.serialize(to_json=True)

    @syft_decorator(typechecking=True)
    def json(self) -> str:
        """A convenience method to convert any subclass of Serializable into a JSON object.

        :return: a JSON string
        :rtype: str
        """
        return self.serialize(to_json=True)

    @syft_decorator(typechecking=True)
    def to_binary(self) -> bytes:
        """A convenience method to convert any subclass of Serializable into a binary object.

        :return: a binary string
        :rtype: bytes
        """
        return self.serialize(to_binary=True)

    @syft_decorator(typechecking=True)
    def binary(self) -> bytes:
        """A convenience method to convert any subclass of Serializable into a binary object.

        :return: a binary string
        :rtype: bytes
        """
        return self.serialize(to_binary=True)

    @syft_decorator(typechecking=True)
    def to_hex(self) -> str:
        """A convenience method to convert any subclass of Serializable into a hex object.

        :return: a hex string
        :rtype: str
        """
        return self.serialize(to_hex=True)

    @syft_decorator(typechecking=True)
    def hex(self) -> str:
        """A convenience method to convert any subclass of Serializable into a hex object.

        :return: a hex string
        :rtype: str
        """
        return self.serialize(to_hex=True)

    @syft_decorator(typechecking=True)
    def serialize(
        self,
        to_proto: bool = True,
        to_json: bool = False,
        to_binary: bool = False,
        to_hex: bool = False,
    ) -> (str, bytes, Message):
        """Serialize the object according to the parameters.

        This is the primary serialization method, which processes the above
        flags in a particular order. In general, it is not expected that people
        will set multiple to_<type> flags to True at the same time. We don't
        currently have logic which prevents this, becuase this may affect
        runtime performance, but if several flags are True, then we will simply
        take return the type of latest supported flag from the following list:

            - proto
            - json
            - binary
            - hex

        TODO: we could also add "dict" to this list but it's not clear if it would be used.

        :param to_proto: set this flag to TRUE if you want to return a protobuf object
        :type to_proto: bool
        :param to_json: set this flag to TRUE if you want to return a json object
        :type to_json: bool
        :param to_binary: set this flag to TRUE if you want to return a binary object
        :type to_binary: bool
        :param to_hex: set this flag to TRUE if you want to return a hex string object
        :type to_hex: bool
        :return: a serialized form of the object on which serialize() is called.
        :rtype: (str,bytes, Message)

        """

        if to_json or to_binary or to_hex:
            blob = json_format.MessageToJson(message=self._object2proto())

            if to_json:
                return blob

            if to_binary or to_hex:
                blob = bytes(blob, "utf-8")
                if to_hex:
                    blob = blob.hex()
            return blob
        elif to_proto:
            return self._object2proto()
        else:
            raise Exception(
                """You must specify at least one deserialization format using
                            one of the arguments of the serialize() method such as:
                            to_proto, to_json, to_binary, or to_hex."""
            )
