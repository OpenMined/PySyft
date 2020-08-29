from abc import ABC
from functools import wraps
from typing import Tuple

from syft.generic.pointers.string_pointer import StringPointer
from syft.generic.string import String


class StringHook(ABC):
    """Hook for ALL THE STRING THINGS that must be overloaded and/or modified"""

    def _hook_string_methods(self, owner):

        # Set the default owner
        setattr(String, "owner", owner)

        for attr in dir(str):

            if attr in String.methods_to_hook:

                # Create the hooked method
                new_method = self._get_hooked_string_method(attr)

                # Add the hooked method
                setattr(String, attr, new_method)

    def _hook_string_pointer_methods(self):

        for attr in dir(String):

            if attr in String.methods_to_hook:

                # Create the hooked method
                new_method = self._get_hooked_string_pointer_method(attr)

                # Add the hooked method
                setattr(StringPointer, attr, new_method)

    @classmethod
    def _string_input_args_adaptor(cls, args_: Tuple[object]):
        """
        This method is used when hooking String methods.

        Some 'String' methods which are overriden from 'str'
        such as the magic '__add__' method
        expects an object of type 'str' as its first
        argument. However, since the '__add__' method
        here is hooked to a String type, it will receive
        arguments of type 'String' not 'str' in some cases.
        This won't worker for the underlying hooked method
        '__add__' of the 'str' type.
        That is why the 'String' argument to '__add__' should
        be peeled down to 'str'

        Args:
            args_: A tuple or positional arguments of the method
                  being hooked to the String class.

        Return:
            A list of adapted positional arguments.

        """

        # If 'arg' is an object of type String
        # replace it by and 'str' object
        return [arg.child if isinstance(arg, String) else arg for arg in args_]

    @classmethod
    def _wrap_str_return_value(cls, _self, attr: str, value: object):

        # The outputs of the following attributed won't
        # be wrapped
        ignored_attr = {"__str__", "__repr__", "__format__"}

        if isinstance(value, str) and attr not in ignored_attr:

            return String(object=value, owner=_self.owner)

        return value

    @classmethod
    def _get_hooked_string_method(cls, attr):
        """
         Hook a `str` method to a corresponding method  of
        `String` with the same name.

         Args:
             attr (str): the method to hook
         Return:
             the hooked method

        """

        @wraps(attr)
        def overloaded_attr(_self, *args, **kwargs):

            args = cls._string_input_args_adaptor(args)

            # Call the method of the core builtin type
            native_response = getattr(_self.child, attr)(*args, **kwargs)

            # Some return types should be wrapped using the String
            # class. For instance, if 'foo' is an object of type
            # 'String' which wraps 'str'. calling foo.upper()
            # should also be of type 'String' not 'str'.
            # However, the return value of foo.__str__ should
            # be of type 'str'.
            response = cls._wrap_str_return_value(_self, attr, native_response)

            return response

        return overloaded_attr

    @classmethod
    def _get_hooked_string_pointer_method(cls, attr):
        """
         Hook a `String` method to a corresponding method  of
        `StringPointer` with the same name.

         Args:
             attr (str): the method to hook
         Return:
             the hooked method

        """

        @wraps(attr)
        def overloaded_attr(_self, *args, **kwargs):
            """
            Operate the hooking
            """

            owner = _self.owner
            location = _self.location
            # id_at_location = self.id_at_location

            # Create a 'command' variable  that is understood by
            # the send_command() method of a worker.
            # command = (attr, id_at_location, args, kwargs)

            # send the command
            response = owner.send_command(location, attr, _self, args, kwargs)

            return response

        return overloaded_attr
