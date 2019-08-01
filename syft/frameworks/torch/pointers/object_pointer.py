import syft
from syft.frameworks.torch.tensors.interpreters import abstract
from syft.codes import MSGTYPE
from syft import exceptions

from typing import List
from typing import Union
from typing import TYPE_CHECKING

# this if statement avoids circular imports between base.py and pointer.py
if TYPE_CHECKING:
    from syft.workers import BaseWorker


class ObjectPointer(abstract.AbstractObject):
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
            id: An optional string or integer id of the PointerTensor.
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

    @classmethod
    def handle_func_command(cls, command):
        """
        Receive an instruction for a function to be applied on a Pointer,
        Get the remote location to send the command, send it and get a
        pointer to the response, return.
        :param command: instruction of a function command: (command name,
        None, arguments[, kwargs])
        :return: the response of the function command
        """
        pointer = cls.find_a_pointer(command)
        # Get info on who needs to send where the command
        owner = pointer.owner
        location = pointer.location

        # Send the command
        response = owner.send_command(location, command)

        return response

    @classmethod
    def find_a_pointer(cls, command):
        """
        Find and return the first pointer in the args object, using a trick
        with the raising error RemoteObjectFoundError
        """
        try:
            cmd, _, args, kwargs = command
            _ = syft.frameworks.torch.hook_args.unwrap_args_from_function(cmd, args, kwargs)
        except exceptions.RemoteObjectFoundError as err:
            pointer = err.pointer
            return pointer

    def get(self, deregister_ptr: bool = True):
        """Requests the object being pointed to.

        The object to which the pointer points will be requested, serialized and returned.

        Note:
            This will typically mean that the remote object will be
            removed/destroyed.

        Args:
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
            obj = self.owner.request_obj(self.id_at_location, self.location)

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
                self.owner.send_msg(MSGTYPE.FORCE_OBJ_DEL, self.id_at_location, self.location)

    def _create_attr_name_string(self, attr_name):
        if self.point_to_attr is not None:
            point_to_attr = "{}.{}".format(self.point_to_attr, attr_name)
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
            message=("__setattr__", self, (name, value), {}), recipient=self.location
        )
