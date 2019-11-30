from typing import List, Tuple
import syft as sy
from syft.generic.pointers.string_pointer import StringPointer
from syft.workers.base import BaseWorker
from syft.generic.object import AbstractObject
import abc


class String(AbstractObject):
    
    # Set of methods from 'str' to hook/override by String
    methods_to_hook = set(
        [
            "__add__", "__eq__", "__le__", "__ge__", "__gt__", "__lt__",  "__ne__",
            "__len__", "__getitem__", "__str__", "__repr__", "__format__", "lower",
            "upper", "capitalize", "casefold", "center", "count", "encode", "endswith",
            "expandtabs", "find", "format", "format_map", "index", "isalnum", "isalpha",
            "isascii", "isdecimal", "isdigit", "isidentifier", "islower", "isnumeric",
            "isprintable", "isspace", "istitle", "isupper", "join", "ljust", "lstrip",
            "maketrans", "partition", "replace", "rfind", "rindex", "rjust", "rpartition",
            "rsplit", "rstrip", "split", "splitlines", "startswith", "strip", "swapcase",
            "title", "translate", "zfill", "__mod__",
            
        ]
    )
    
    def __init__(
        self,
        object: object = None,
        encoding: str = None,
        errors: str = None,
        id: int = None,
        owner: BaseWorker = None,
        tags: List[str] = None,
        description: str = None,
    ):
        # get the specified kwargs for creating the base 'str'
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

        def simplify_str(str):
            return str.encode("utf-8") if str else None

        # Encode the string into a bytes object
        simple_core = simplify_str(string.child)

        tags = (
            [simplify_str(tag) if isinstance(tag, str) else tag for tag in string.tags]
            if string.tags
            else None
        )
        description = simplify_str(string.description)

        return (simple_core, string.id, tags, description)

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
        simple_core, id, tags, description = simple_obj

        def detail_bstr(b_str):
            return str(b_str, encoding="utf-8") if b_str else None

        # It appears that all strings are converted to bytes objects
        # after deserialization, convert them back to strings
        tags = (
            [detail_bstr(tag) if isinstance(tag, bytes) else tag for tag in tags] if tags else None
        )
        description = detail_bstr(description)

        # Rebuild the str core our of the simplified core (the bytes core)
        core = str(simple_core, encoding="utf-8", errors="strict")

        return String(object=core, id=id, owner=worker, tags=tags, description=description)
