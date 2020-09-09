from typing import List
from typing import Union
from typing import TYPE_CHECKING
import weakref
from websocket._exceptions import WebSocketConnectionClosedException

import syft
from syft import exceptions
from syft.generic.frameworks.hook.hook_args import one
from syft.generic.frameworks.hook.hook_args import register_type_rule
from syft.generic.frameworks.hook.hook_args import register_forward_func
from syft.generic.frameworks.hook.hook_args import register_backward_func
from syft.generic.frameworks.hook import hook_args
from syft.generic.frameworks.types import FrameworkTensor
from syft.generic.abstract.sendable import AbstractSendable
from syft.workers.abstract import AbstractWorker

from syft.exceptions import RemoteObjectFoundError

# this if statement avoids circular imports between base.py and pointer.py
if TYPE_CHECKING:
    from syft.workers.base import BaseWorker


class ObjectPointer(AbstractSendable):
    """A pointer to a remote object.

    An ObjectPointer forwards all API calls to the remote. ObjectPointer objects
    point to objects. They exist to mimic the entire
    API of an object, but instead of computing a function locally
    (such as addition, subtraction, etc.) they forward the computation to a
    remote machine as specified by self.location. Specifically, every
    ObjectPointer has a object located somewhere that it points to (they should
    never exist by themselves).
    The objects being pointed to can be on the same machine or (more commonly)
    on a different one. Note further that a ObjectPointer does not know the
    nature how it sends messages to the object it points to (whether over
    socket, http, or some other protocol) as that functionality is abstracted
    in the BaseWorker object in self.location.
    """

    def __init__(
        self,
        location: "BaseWorker" = None,
        id_at_location: Union[str, int] = None,
        owner: "BaseWorker" = None,
        id: Union[str, int] = None,
        garbage_collect_data: bool = True,
        point_to_attr: str = None,
        tags: List[str] = None,
        description: str = None,
    ):
        """Initializes a ObjectPointer.

        Args:
            location: An optional BaseWorker object which points to the worker
                on which this pointer's object can be found.
            id_at_location: An optional string or integer id of the object
                being pointed to.
            owner: An optional BaseWorker object to specify the worker on which
                the pointer is located. It is also where the pointer is
                registered if register is set to True. Note that this is
                different from the location parameter that specifies where the
                pointer points to.
            id: An optional string or integer id of the ObjectPointer.
            garbage_collect_data: If true (default), delete the remote object when the
                pointer is deleted.
            point_to_attr: string which can tell a pointer to not point directly to\
                an object, but to point to an attribute of that object such as .child or
                .grad. Note the string can be a chain (i.e., .child.child.child or
                .grad.child.child). Defaults to None, which means don't point to any attr,
                just point to then object corresponding to the id_at_location.
        """
        super().__init__(id=id, owner=owner, tags=tags, description=description)

        self.location = location
        self.id_at_location = id_at_location
        self.garbage_collect_data = garbage_collect_data
        self.point_to_attr = point_to_attr

    @staticmethod
    def create_pointer(
        obj,
        location: "AbstractWorker" = None,
        id_at_location: (str or int) = None,
        register: bool = False,
        owner: "AbstractWorker" = None,
        ptr_id: (str or int) = None,
        garbage_collect_data=None,
    ) -> "ObjectPointer":
        """Creates a pointer to the "self" FrameworkTensor object.

        This method is called on a FrameworkTensor object, returning a pointer
        to that object. This method is the CORRECT way to create a pointer,
        and the parameters of this method give all possible attributes that
        a pointer can be created with.

        Args:
            location: The AbstractWorker object which points to the worker on which
                this pointer's object can be found. In nearly all cases, this
                is self.owner and so this attribute can usually be left blank.
                Very rarely you may know that you are about to move the Tensor
                to another worker so you can pre-initialize the location
                attribute of the pointer to some other worker, but this is a
                rare exception.
            id_at_location: A string or integer id of the tensor being pointed
                to. Similar to location, this parameter is almost always
                self.id and so you can leave this parameter to None. The only
                exception is if you happen to know that the ID is going to be
                something different than self.id, but again this is very rare
                and most of the time, setting this means that you are probably
                doing something you shouldn't.
            register: A boolean parameter (default False) that determines
                whether to register the new pointer that gets created. This is
                set to false by default because most of the time a pointer is
                initialized in this way so that it can be sent to someone else
                (i.e., "Oh you need to point to my tensor? let me create a
                pointer and send it to you" ). Thus, when a pointer gets
                created, we want to skip being registered on the local worker
                because the pointer is about to be sent elsewhere. However, if
                you are initializing a pointer you intend to keep, then it is
                probably a good idea to register it, especially if there is any
                chance that someone else will initialize a pointer to your
                pointer.
            owner: A AbstractWorker parameter to specify the worker on which the
                pointer is located. It is also where the pointer is registered
                if register is set to True.
            ptr_id: A string or integer parameter to specify the id of the pointer
                in case you wish to set it manually for any special reason.
                Otherwise, it will be set randomly.
            garbage_collect_data: If true (default), delete the remote tensor when the
                pointer is deleted.
            local_autograd: Use autograd system on the local machine instead of PyTorch's
                autograd on the workers.
            preinitialize_grad: Initialize gradient for AutogradTensors to a tensor.

        Returns:
            A FrameworkTensor[ObjectPointer] pointer to self. Note that this
            object itself will likely be wrapped by a FrameworkTensor wrapper.
        """
        if owner is None:
            owner = obj.owner

        if location is None:
            location = obj.owner.id

        owner = obj.owner.get_worker(owner)
        location = obj.owner.get_worker(location)

        ptr = ObjectPointer(
            location=location,
            id_at_location=id_at_location,
            owner=owner,
            id=ptr_id,
            garbage_collect_data=True if garbage_collect_data is None else garbage_collect_data,
            tags=obj.tags,
            description=obj.description,
        )

        return ptr

    def wrap(self, register=True, type=None, **kwargs):
        """Wraps the class inside framework tensor.

        Because PyTorch/TF do not (yet) support functionality for creating
        arbitrary Tensor types (via subclassing torch.Tensor), in order for our
        new tensor types (such as PointerTensor) to be usable by the rest of
        PyTorch/TF (such as PyTorch's layers and loss functions), we need to
        wrap all of our new tensor types inside of a native PyTorch type.

        This function adds a .wrap() function to all of our tensor types (by
        adding it to AbstractTensor), such that (on any custom tensor
        my_tensor), my_tensor.wrap() will return a tensor that is compatible
        with the rest of the PyTorch/TensorFlow API.

        Returns:
            A wrapper tensor of class `type`, or whatever is specified as
            default by the current syft.framework.Tensor.
        """
        wrapper = syft.framework.hook.create_wrapper(type, **kwargs)
        wrapper.child = self
        wrapper.is_wrapper = True
        wrapper.child.parent = weakref.ref(wrapper)

        if self.id is None:
            self.id = syft.ID_PROVIDER.pop()

        if self.owner is not None and register:
            self.owner.register_obj(wrapper, obj_id=self.id)

        return wrapper

    @classmethod
    def handle_func_command(cls, command):
        """
        Receive an instruction for a function to be applied on a Pointer,
        Get the remote location to send the command, send it and get a
        pointer to the response, return.
        :param command: instruction of a function command: (command name,
        None, arguments[, kwargs_])
        :return: the response of the function command
        """
        pointer = cls.find_a_pointer(command)
        # Get info on who needs to send where the command
        owner = pointer.owner
        location = pointer.location

        cmd, _, args_, kwargs_ = command

        # Send the command
        response = owner.send_command(location, cmd_name=cmd, args_=args_, kwargs_=kwargs_)

        return response

    @classmethod
    def find_a_pointer(cls, command):
        """
        Find and return the first pointer in the args_ object, using a trick
        with the raising error RemoteObjectFoundError
        """
        try:
            cmd, _, args_, kwargs_ = command
            _ = hook_args.unwrap_args_from_function(cmd, args_, kwargs_)
        except exceptions.RemoteObjectFoundError as err:
            pointer = err.pointer
            return pointer

    def get(self, user=None, reason: str = "", deregister_ptr: bool = True):
        """Requests the object being pointed to.

        The object to which the pointer points will be requested, serialized and returned.

        Note:
            This will typically mean that the remote object will be
            removed/destroyed.

        Args:
            user (obj, optional) : authenticate/allow user to perform get on remote private objects.
            reason (str, optional) : a description of why the data scientist wants to see it.
            deregister_ptr (bool, optional): this determines whether to
                deregister this pointer from the pointer's owner during this
                method. This defaults to True because the main reason people use
                this method is to move the tensor from the location to the
                local one, at which time the pointer has no use.

        Returns:
            An AbstractObject object which is the tensor (or chain) that this
            object used to point to on a location.

        TODO: add param get_copy which doesn't destroy remote if true.
        """

        if self.point_to_attr is not None:

            raise exceptions.CannotRequestObjectAttribute(
                "You called .get() on a pointer to"
                " a tensor attribute. This is not yet"
                " supported. Call .clone().get() instead."
            )

        # if the pointer happens to be pointing to a local object,
        # just return that object (this is an edge case)
        if self.location == self.owner:
            obj = self.owner.get_obj(self.id_at_location)
            if hasattr(obj, "child"):
                obj = obj.child
        else:
            # get tensor from location
            obj = self.owner.request_obj(self.id_at_location, self.location, user, reason)

        # Remove this pointer by default
        if deregister_ptr:
            self.owner.de_register_obj(self)

        if self.garbage_collect_data:
            # data already retrieved, do not collect any more.
            self.garbage_collect_data = False

        return obj

    def __str__(self):
        """Returns a string version of this pointer.

        This is primarily for end users to quickly see things about the object.
        This tostring shouldn't be used for anything else though as it's likely
        to change. (aka, don't try to parse it to extract information. Read the
        attribute you need directly). Also, don't use this to-string as a
        serialized form of the pointer.
        """

        type_name = type(self).__name__
        out = (
            f"["
            f"{type_name} | "
            f"{str(self.owner.id)}:{self.id}"
            " -> "
            f"{str(self.location.id)}:{self.id_at_location}"
            f"]"
        )

        if self.point_to_attr is not None:
            out += "::" + str(self.point_to_attr).replace(".", "::")

        big_str = False

        if self.tags is not None and len(self.tags):
            big_str = True
            out += "\n\tTags: "
            for tag in self.tags:
                out += str(tag) + " "

        if big_str and hasattr(self, "shape"):
            out += "\n\tShape: " + str(self.shape)

        if self.description is not None:
            big_str = True
            out += "\n\tDescription: " + str(self.description).split("\n")[0] + "..."

        return out

    def __repr__(self):
        """Returns the to-string method.

        When called using __repr__, most commonly seen when returned as cells
        in Jupyter notebooks.
        """
        return self.__str__()

    def __del__(self):
        """This method garbage collects the object this pointer is pointing to.
        By default, PySyft assumes that every object only has one pointer to it.
        Thus, if the pointer gets garbage collected, we want to automatically
        garbage collect the object being pointed to.
        """

        # if .get() gets called on the pointer before this method is called, then
        # the remote object has already been removed. This results in an error on
        # this next line because self no longer has .owner. Thus, we need to check
        # first here and not try to call self.owner.anything if self doesn't have
        # .owner anymore.
        if hasattr(self, "owner") and self.garbage_collect_data:
            # attribute pointers are not in charge of GC
            if self.point_to_attr is None:
                try:
                    self.owner.garbage(self.id_at_location, self.location)
                except (BrokenPipeError, WebSocketConnectionClosedException):
                    pass

    def _create_attr_name_string(self, attr_name):
        if self.point_to_attr is not None:
            point_to_attr = f"{self.point_to_attr}.{attr_name}"
        else:
            point_to_attr = attr_name
        return point_to_attr

    def attr(self, attr_name):
        attr_ptr = syft.ObjectPointer(
            id=self.id,
            owner=self.owner,
            location=self.location,
            id_at_location=self.id_at_location,
            point_to_attr=self._create_attr_name_string(attr_name),
        )  # .wrap()
        self.__setattr__(attr_name, attr_ptr)
        return attr_ptr

    def setattr(self, name, value):
        self.owner.send_command(
            cmd_name="__setattr__",
            target=self,
            args_=(name, value),
            kwargs_={},
            recipient=self.location,
        )

    @staticmethod
    def simplify(worker: AbstractWorker, ptr: "ObjectPointer") -> tuple:
        """
        This function takes the attributes of a ObjectPointer and saves them in a dictionary
        Args:
            ptr (ObjectPointer): a ObjectPointer
        Returns:
            tuple: a tuple holding the unique attributes of the pointer
        Examples:
            data = simplify(ptr)
        """

        return (
            syft.serde.msgpack.serde._simplify(worker, ptr.id),
            syft.serde.msgpack.serde._simplify(worker, ptr.id_at_location),
            syft.serde.msgpack.serde._simplify(worker, ptr.location.id),
            syft.serde.msgpack.serde._simplify(worker, ptr.point_to_attr),
            ptr.garbage_collect_data,
        )

    @staticmethod
    def detail(worker: "AbstractWorker", object_tuple: tuple) -> "ObjectPointer":
        """
        This function reconstructs an ObjectPointer given it's attributes in form of a dictionary.
        We use the spread operator to pass the dict data as arguments
        to the init method of ObjectPointer
        Args:
            worker: the worker doing the deserialization
            tensor_tuple: a tuple holding the attributes of the ObjectPointer
        Returns:
            ObjectPointer: an ObjectPointer
        Examples:
            ptr = detail(data)
        """
        # TODO: fix comment for this and simplifier
        obj_id, id_at_location, worker_id, point_to_attr, garbage_collect_data = object_tuple

        obj_id = syft.serde.msgpack.serde._detail(worker, obj_id)
        id_at_location = syft.serde.msgpack.serde._detail(worker, id_at_location)
        worker_id = syft.serde.msgpack.serde._detail(worker, worker_id)
        point_to_attr = syft.serde.msgpack.serde._detail(worker, point_to_attr)

        # If the pointer received is pointing at the current worker, we load the tensor instead
        if worker_id == worker.id:
            obj = worker.get_obj(id_at_location)

            if point_to_attr is not None and obj is not None:

                point_to_attrs = point_to_attr.split(".")
                for attr in point_to_attrs:
                    if len(attr) > 0:
                        obj = getattr(obj, attr)

                if obj is not None:

                    if not obj.is_wrapper and not isinstance(obj, FrameworkTensor):
                        # if the object is a wrapper then it doesn't need to be wrapped
                        # i the object isn't a wrapper, BUT it's just a plain torch tensor,
                        # then it doesn't need to be wrapped.
                        # if the object is not a wrapper BUT it's also not a framework object,
                        # then it needs to be wrapped or else it won't be able to be used
                        # by other interfaces
                        obj = obj.wrap()

            return obj
        # Else we keep the same Pointer
        else:

            location = syft.hook.local_worker.get_worker(worker_id)

            ptr = ObjectPointer(
                location=location,
                id_at_location=id_at_location,
                owner=worker,
                id=obj_id,
                garbage_collect_data=garbage_collect_data,
            )

            return ptr


### Register the object with hook_args.py ###
register_type_rule({ObjectPointer: one})
register_forward_func({ObjectPointer: lambda p: (_ for _ in ()).throw(RemoteObjectFoundError(p))})
register_backward_func({ObjectPointer: lambda i: i})
