from typing import List
from typing import Union

from syft.generic.string import String
from syft.generic.pointers.object_pointer import ObjectPointer
from syft.workers.base import BaseWorker

import abc

class PointerClassMaker(abc.ABCMeta):
    """
       This is a meta class.

       It is used to create StringPointer and hook all of the
       methods in String to it. Then the functionality of those
       methods will be overriden.
    """

    def __new__(cls,
                clsname,
                bases,
                dct,
                pointed_type):


        # Set of magic method to hook/override in case they exist
        magic_attr = set(['__add__',
                          '__eq__',
                          '__le__',
                          '__ge__',
                          '__gt__',
                          '__lt--',
                          '__ne__',
                          '__len__',
                          '__getitem__'
        ])

        # Set of non-magice methods that should not be hooked and overriden
        # TODO: This list can be created partially automatically by
        # excluding all methods coming from 'bases'
        ignore_attr = set(['send',
                           'get',
                           'simplify',
                           'detail',
                           'create_pointer',
        ])


        for attr in dir(pointed_type):

            # flag whether to hook the current attribute or not
            hook = False
            
            # Get only magic attributes that are not
            # in the 'ignored_attr' set
            if attr.startswith('__'):
                
                hook = True if attr in magic_attr else False
                
            elif attr not in ignore_attr:
                hook = True

            if hook:
                dct[attr] = cls.method_hook(attr)


        return super(PointerClassMaker, cls).__new__(cls, clsname, bases, dct)


    def method_hook(attr: str
    ):
        """
           This method creates and returns a function
           that defines the functionality of the the method named
           'attr' of the class being built by the current meta class.
           Each such method forwards the calls to the corresponding
           method with the same name of the object of type 'hooked_type'
           That the underlying class is pointer to.
           
           Args:
               cls: A meta class object representing the current class
               attr: A 'str' object representing the attribute name
                     that is being hooked.

           Return:
              A Callable that will be used as the attribute 'attr'
              of the new wrapping class.
        """

        def overloaded_pointer_method(pointed_obj, *args, **kwargs):

            owner = pointed_obj.owner
            location = pointed_obj.location
            id_at_location = pointed_obj.id_at_location
            # Create a 'command' variable  that is understood by
            # the send_command() method of a worker.
            command = (attr, id_at_location, args, kwargs)

            # send the command
            response = owner.send_command(location, command)
            
            return response

        return overloaded_pointer_method
    
class StringPointer(ObjectPointer, metaclass = PointerClassMaker, pointed_type = String):
    """
       This class defines a pointer to a 'String' object that might live
       on a remote machine. In other words, it holds a pointer to a 
       'String' object owned by a possibly different worker (although 
       it can also point to a String owned by the same worker'.

       All String method are hooked to objects of this class, and calls to 
       such methods are forwarded to the pointed-to String object.
    """
    
    def __init__(
        self,
        location: BaseWorker = None,
        id_at_location: Union[str, int] = None,
        owner: BaseWorker = None,
        id: Union[str, int] = None,
        garbage_collect_data: bool = True,
        tags: List[str] = None,
        description: str = None,
    ):

        super(StringPointer, self).__init__(location = location,
                                            id_at_location = id_at_location,
                                            owner = owner,
                                            id = id,
                                            garbage_collect_data = garbage_collect_data,
                                            tags = tags,
                                            description = description)
                                    
                                        
    
