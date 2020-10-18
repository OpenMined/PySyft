from typing import List
from typing import Tuple
from typing import Union
import syft as sy
from syft.generic.pointers.string_pointer import StringPointer
from syft.workers.base import BaseWorker
from syft.generic.abstract.sendable import AbstractSendable
from syft.generic.frameworks.overload import overloaded
from syft.generic.frameworks.hook import hook_args
from syft_proto.generic.string_pb2 import String as StringPB


class String(AbstractSendable):
    """
       This is a class that wraps the Python built-in `str` class. In addition to
    providing access to most of `str`'s method call API, it allows sending
    such wrapped string between workers the same way Syft tensors can be
    moved around among workers.
    """

    # Set of methods from 'str' to hook/override by String
    methods_to_hook = {
        "__add__",
        "__eq__",
        "__le__",
        "__ge__",
        "__gt__",
        "__lt__",
        "__ne__",
        "__len__",
        "__getitem__",
        "__str__",
        "__repr__",
        "__format__",
        "lower",
        "upper",
        "capitalize",
        "casefold",
        "center",
        "count",
        "encode",
        "endswith",
        "expandtabs",
        "find",
        "format",
        "format_map",
        "index",
        "isalnum",
        "isalpha",
        "isascii",
        "isdecimal",
        "isdigit",
        "isidentifier",
        "islower",
        "isnumeric",
        "isprintable",
        "isspace",
        "istitle",
        "isupper",
        "join",
        "ljust",
        "lstrip",
        "maketrans",
        "partition",
        "replace",
        "rfind",
        "rindex",
        "rjust",
        "rpartition",
        "rsplit",
        "rstrip",
        "split",
        "splitlines",
        "startswith",
        "strip",
        "swapcase",
        "title",
        "translate",
        "zfill",
        "__mod__",
    }

    def __init__(
        self,
        object: object = None,
        encoding: str = None,
        errors: str = None,
        id: Union[int, str] = None,
        owner: BaseWorker = None,
        tags: List[str] = None,
        description: str = None,
    ):
        """Initialize a String object.

        Args:
           object: This could be any object whose string representation,i.e.,
               the output of its __str__() method is to be wrapped as a
               String object.
           encoding: This should be specified if the above `object` argument is
               a bytes-like object. It specifies the encoding scheme used to create the
               bytes-like object b''. for example, encoding could be 'utf-8'.
               For more details on this argument, please  check the official `str`
               documentation.
           errors: This should be specified if the above `object` argument is
               a bytes-like object. Possible values are 'strict', 'ignore' or
               'replace'. For more details on this argument, please
               check the official `str` documentation.
           id: An optional string or integer id of the String object
           owner: An optional BaseWorker object to specify the worker on which
               the String object is located.
           tags: an optional set of hashtags corresponding to this object.
               They are useful when search for this object.
           description: an optional string describing the purpose of this
               String object

        """

        # get the specified kwargs_ for creating the base 'str'
        # class

        self.encoding = encoding
        self.errors = errors

        # String objects have normally a default owner which is the
        # local worker. So prevent 'None' as owner
        if self.owner is None or owner is not None:
            self.owner = owner

        str_kwargs = {}

        if object:
            str_kwargs["object"] = object

        if encoding:
            str_kwargs["encoding"] = encoding

        if errors:
            str_kwargs["errors"] = errors

        # Create a str instance as the 'child' attribute
        self.child = str(**str_kwargs)

        super(String, self).__init__(
            id=id, owner=self.owner, tags=tags, description=description, child=self.child
        )

    def send(self, location: BaseWorker):
        """
           Sends this String object to the worker specified by 'location'.
        and returns a pointer to that string as a StringPointer object.

        Args:
           location: The BaseWorker object which you want to send this object
                     to. Note that this is never actually the BaseWorker but instead
                     a class which inherits the BaseWorker abstraction.

        Returns:
           A StringPointer objects to self.
        """

        ptr = self.owner.send(self, location)

        return ptr

    def get_class_attributes(self):
        """
        Returns: minimal necessary keyword arguments to create a
           String object
        """
        kwargs_ = {"owner": self.owner}

        return kwargs_

    def on(self, object: str, wrap=False):
        """Takes and object of type strings and assigns it to
        self.child
        """
        self.child = object

        return self

    def __add__(self, other: Union[str, "String"]):
        """[Important] overriding the `__add__` here is not yet
        activated. The real hooking happens in
        syft/generic/frameworks/hook/hook.py.
        Hooking as implemented here (using @overloaded.method)
        is to be activated when hook_args.py is adapted
        to wrapping reponses of `str` types into `String`
        types. This is not yet supported.
        """

        # The following is necessary in order to adapt the
        # below `add_string` method to the args hooking logic in
        # hook_args.py. Please check the doc string of `add_string`
        # to know more.
        if isinstance(other, str):
            other = String(other)

        return self.add_string(other)

    @overloaded.method
    def add_string(self, _self: "String", other: "String"):
        """This method is created in a way adapted to the logic implemented
        in hook_args.py. That is, it can be wrapped with the decorator
        @overloaded.method.

        hook_args.py args hooking logic needs that the data types of
        argument be unchanged. For instance, 'other' should always
        be of a fixed type 'String' or 'str' but not alternating
        between both. This can cause unexpected behaviou due to caching
        in hook_args.py.

        Args:
           _self: a String object (as received by the decorator).
                  It represents the objects on which we called the add method.
                  It will always be of type `str` inside this method. Since
                  the decorator methods strips the `str` out of the `String`
                  object.
           other: a String object that we wish to concatenate to `_self`.
                  Same as above, it is a String object as received by the
                  decorator but here it will always be of type `str`.

        Returns:
           The concatentenated `str` object between `_self` and `other`.
           this `str` object will be wrapped by the decorator into a
           String object
        """

        return _self + other

    @staticmethod
    def create_pointer(
        obj,
        location: BaseWorker = None,
        id_at_location: (str or int) = None,
        register: bool = False,
        owner: BaseWorker = None,
        ptr_id: (str or int) = None,
        garbage_collect_data: bool = True,
    ):
        """
        Creates a StringPointer object that points to a String object 'obj'
        after sending the latter to the worker 'location'.

        Returns:
            a StringPointer object
        """

        if id_at_location is None:
            id_at_location = obj.id

        if owner is None:
            owner = obj.owner

        string_pointer = StringPointer(
            location=location,
            id_at_location=id_at_location,
            owner=owner,
            id=ptr_id,
            garbage_collect_data=garbage_collect_data,
        )

        return string_pointer

    @staticmethod
    def simplify(worker: BaseWorker, string: "String"):
        """
        Breaks String object into a tuple of simpler objects, its constituting objects that are
        serializable.

        Args:
           worker: a BaseWorker object
           string: the String object to be simplified

         Returns:
           A tuple of simpler objects that are sufficient to recreate
           a String object that is a clone of `string`.
        """

        # Encode the string into a bytes object
        simple_child = sy.serde.msgpack.serde._simplify(worker, string.child)
        tags = sy.serde.msgpack.serde._simplify(worker, string.tags)
        description = sy.serde.msgpack.serde._simplify(worker, string.description)

        return (simple_child, string.id, tags, description)

    @staticmethod
    def detail(worker: BaseWorker, simple_obj: Tuple):
        """
        Create an object of type String from the reduced representation in `simple_obj`.


        Args:
           worker: BaseWorker
                   The worker on which the new String object is to be created.
           simple_obj: tuple
                       A tuple resulting from the serialized then deserialized returned tuple
                       from the `simplify` static method above.

        Returns:
           A String object
        """

        # Get the contents of the tuple represening the simplified object
        simple_child, id, tags, description = simple_obj

        # It appears that all strings are converted to bytes objects
        # after deserialization, convert them back to strings
        tags = sy.serde.msgpack.serde._detail(worker, tags)
        description = sy.serde.msgpack.serde._detail(worker, description)

        # Rebuild the str child our of the simplified child (the bytes child)
        child = sy.serde.msgpack.serde._detail(worker, simple_child)

        return String(object=child, id=id, owner=worker, tags=tags, description=description)

    @staticmethod
    def bufferize(worker, str_object):
        """
        This method serializes a String into a StringPB.

            Args:
                str_object (String): input String to be serialized.

            Returns:
                proto_string (StringPB): serialized String.
        """
        proto_string = StringPB()
        proto_string.child = str_object.child
        for tag in str_object.tags:
            proto_string.tags.append(tag)
        if str_object.description:
            proto_string.description = str_object.description

        return proto_string

    @staticmethod
    def unbufferize(worker, obj):
        """
        This method deserializes StringPB into a String.

        Args:
            obj (StringPB): input serialized StringPB.

        Returns:
            String: deserialized ScriptFunctionPB.
        """
        return String(object=obj.child, tags=obj.tags, description=obj.description)

    @staticmethod
    def get_protobuf_schema():
        """
        This method returns the protobuf schema used for String.

        Returns:
           Protobuf schema for String.
        """
        return StringPB


### Register the String object with hook_args.py ###
hook_args.default_register_tensor(String)
