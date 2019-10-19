from typing import List, Tuple
import syft as sy
from syft.workers.base import BaseWorker
from syft.generic.object import AbstractObject
import abc

class BuiltinTypeWrapper(abc.ABCMeta):
    """
       This is a meta class.

       It is used to create a wrapper class around a python
       built-in type such as 'str', 'list', 'dict'.
    """

    def __new__(cls,
                clsname,
                bases,
                dct,
                wrapped_type: type):

        
        # Set of attributes to ignore by the wrapper.
        # Those attributes are not to be overridden.
        ignored_attr = set(['__class__',
                            '__mro__',
                            '__new__',
                            '__init__',
                            '__init_subclass__',
                            '__subclasshook__',
                            '__setattr__',
                            '__getattr__',
                            '__getattribute__',
        ])
        

        
        for attr in dir(wrapped_type):

            # Get only magic attributes that are not
            # in the 'ignored_attr' set
            if attr not in ignored_attr:

                dct[attr] = cls.method_hook(cls, attr)

        return super(BuiltinTypeWrapper, cls).__new__(cls, clsname, bases, dct)

    
    def _args_adaptor(self: object,
                      args: Tuple[object]
    ):
        """
           Adapt the 'args' tuple content types so that
           it can be consumed by native methods of
           the built-in type in 'self.core'.
           For instance, when the built-in type in
           'self.core' is 'str', the magic '__add__' method
           expects an object of type 'str' as its first
           argument. However, since the '__add__' method
           here is hooked to a wrapper type, for instance
           String which is the PySyft type that wraps 'str'
           and hooks all of its methods, there is a risk
           that the first argument of the method '__add__'
           be of that wrapper type. So it cannot be
           consumed by the native 'str' method '__add__'
           when calling:
           
           > getattr(self.core, attr)(*args, **kwargs)

           Args:
               self: An object of the class that is created
                     by this __metaclass__. In other words, this
                     is an object of the class type wrapping around
                     a python built-in type.
               args: A tuple or positional arguments of the method
                     being hooked to the wrapper class.
                     Check out the above '__new__' method to see
                     where 'args' appear.

           Returns:
               A list of adapted positional arguments.
           
        """

        new_args = []
        
        for arg in args:

            # If 'arg' is an object of the class
            # wrapping around the built-in type,
            # replace it by the wrapped built-in object.
            if isinstance(arg, self.__class__):
                new_args.append(arg.core)
            else:
                new_args.append(arg)

        return new_args

    
    def method_hook(cls,
                    attr: str
    ):
        """
           This method creates and returns a function
           that will be hooked to the wrapper class
           being built by the current meta class.
           This method will take the name 'attr' in the
           wrapper class. It wraps the corresponding
           method with the same name of the underlying
           built-in type wrapped by the wrapper class.
           
           Args:
               cls: A meta class object representing the current class
               attr: A 'str' object representing the attribute name
                     that is being hooked.

           Return:
              A Callable that will be used as the attribute 'attr'
              of the new wrapping class.
        """

        def wrapper(self, *args, **kwargs):

            args = cls._args_adaptor(self, args)

            # Call the method of the core builtin type
            native_output = getattr(self.core, attr)(*args, **kwargs)

            # Some return types should be wrapped using an
            # object of PySyft built-in-wrapper classes.
            # For instance, if 'foo' is an object of type
            # 'String' which wraps 'str'. calling foo.upper()
            # should also be of type 'String' not 'str'.
            # However, the return value of foo.__str__ should
            # be of type 'str'.
            output = cls._wrap_return_value(self, attr, native_output)
            
            return output

        return wrapper


    def _wrap_return_value(self,
                           attr: str,
                           value: object
    ):

        # The outputs of the following attributed won't
        # be wrapped
        ignored_attr = set(['__str__',
                            '__repr__',
                            '__format__'
        ])
        
        if isinstance(value, str) and attr not in ignored_attr:
            #TODO: change the __init__ call here
            return String(object = value,
                          owner = self.owner)

        return value


class String(AbstractObject, metaclass = BuiltinTypeWrapper, wrapped_type = str):


    def __init__(self,
                object: object = None,
                encoding: str = None,
                errors: str = None,
                id: int = None,
                owner: BaseWorker = None,
                tags: List[str] = None,
                description: str = None
                
    ):
        # get the specified kwargs for creating the base 'str'
        # class

        self.encoding = encoding
        self.errors = errors

        str_kwargs = {}
        
        if object:
            str_kwargs['object'] = object

        if encoding:
            str_kwargs['encoding'] = encoding

        if errors:
            str_kwargs['errors'] = errors

        # Create a str instance as the 'child' attribute
        self.core = str(**str_kwargs)

        super(String, self).__init__(id = id,
                                     owner = owner,
                                     tags = tags,
                                     description = description)        



    def send(self,
             location: BaseWorker
    ):
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

        ptr = self.owner.send(
            self,
            location
        )

        return ptr


    @staticmethod
    def create_pointer(obj,
                       location: BaseWorker = None,
                       id_at_location: (str or int) = None,
                       register : bool = False,
                       owner: BaseWorker = None,
                       ptr_id: (str or int) = None,
                       garbage_collect_data: bool = True,
    ) -> "StringPointer":
        """
           Creates a StringPointer object that points to a String object 'obj'
           after sending the latter to the worker 'location'.
        """

        # I put the import here in order to avoid circular imports
        # between string_pointer.py and this file (string.py).
        # In order to get rid of this local import (which I do not like)
        # I think that the 'create_pointer' functionality should be
        # changed. We can create a separated class called 'PointerMaker'
        # that lives in its own file. It creates pointers to differenct
        # types of objects.
        # Actually, there is no strong reason why the 'create_pointer'
        # method should be defined inside the String class. It is
        # a static method anyway.
        from syft.generic.pointers.string_pointer import StringPointer

        if id_at_location is None:
            id_at_location = obj.id

        if owner is None:
            owner = obj.owner
            
        string_pointer =  StringPointer(location = location,
                                        id_at_location = id_at_location,
                                        owner = owner,
                                        id = ptr_id,
                                        garbage_collect_data = garbage_collect_data)

        return string_pointer
    
    @staticmethod
    def simplify(string: "String"):
        """
           TODO:
           add docstring
        """

        def simplify_str(str):
            return str.encode('utf-8') if str else None

        # Encode the string into a bytes object
        simple_core = simplify_str(string.core)
        
        tags = [simplify_str(tag) if isinstance(tag, str) else tag for tag in string.tags] if string.tags else None
        description = simplify_str(string.description)
        
        return (simple_core,
                string.id,
                tags,
                description)

    @staticmethod
    def detail(worker: BaseWorker,
               simple_obj: Tuple
    ):
        """
           Create an object of type String from the reduced representation in `simple_obj`.

           Parameters
           ----------
           worker: BaseWorker
                   The worker on which the new String object is to be created.
           simple_obj: tuple
                       A tuple resulting from the serialized then deserialized returned tuple
                       from the `simplify` static method above.

           Returns
           -------
           A String object


        """
        
        # Get the contents of the tuple represening the simplified object
        simple_core, id, tags, description = simple_obj        

        def detail_bstr(b_str):
            return str(b_str, encoding = 'utf-8') if b_str else None
        
        # It appears that all strings are converted to bytes objects
        # after deserialization, convert them back to strings
        tags = [detail_bstr(tag) if isinstance(tag, bytes) else tag for tag in tags] if tags else None
        description = detail_bstr(description)

        # Rebuild the str core our of the simplified core (the bytes core)
        core = str(simple_core, encoding = 'utf-8', errors = 'strict')        

        return String(object = core,
                      id = id,
                      owner = worker,
                      tags = tags,
                      description = description
        )



