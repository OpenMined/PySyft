import syft
import torch
from syft.frameworks.torch.tensors.interpreters.abstract import AbstractTensor
from syft.codes import MSGTYPE
from syft.exceptions import RemoteTensorFoundError, CannotRequestTensorAttribute


class PointerTensor(AbstractTensor):
    """A pointer to another tensor.

    A PointerTensor forwards all API calls to the remote.PointerTensor objects
    point to tensors (as their name implies). They exist to mimic the entire
    API of a normal tensor, but instead of computing a tensor function locally
    (such as addition, subtraction, etc.) they forward the computation to a
    remote machine as specified by self.location. Specifically, every
    PointerTensor has a tensor located somewhere that it points to (they should
    never exist by themselves). Note that PointerTensor objects can point to
    both torch.Tensor objects AND to other PointerTensor objects. Furthermore,
    the objects being pointed to can be on the same machine or (more commonly)
    on a different one. Note further that a PointerTensor does not know the
    nature how it sends messages to the tensor it points to (whether over
    socket, http, or some other protocol) as that functionality is abstracted
    in the BaseWorker object in self.location.

    Example:

     >>> import syft as sy
     >>> hook = sy.TorchHook()
     >>> bob = sy.VirtualWorker(id="bob")
     >>> x = sy.Tensor([1,2,3,4,5])
     >>> y = sy.Tensor([1,1,1,1,1])
     >>> x_ptr = x.send(bob) # returns a PointerTensor, sends tensor to Bob
     >>> y_ptr = y.send(bob) # returns a PointerTensor, sends tensor to Bob
     >>> # executes command on Bob's machine
     >>> z_ptr = x_ptr + y_ptr
    """

    def __init__(
        self,
        parent: AbstractTensor = None,
        location=None,
        id_at_location=None,
        register=False,
        owner=None,
        id=None,
        garbage_collect_data=True,
        shape=None,
        point_to_attr=None,
        tags=None,
        description=None,
    ):
        """Initializes a PointerTensor.

        Args:
            parent: An optional AbstractTensor wrapper around the PointerTensor
                which makes it so that you can pass this PointerTensor to all
                the other methods/functions that PyTorch likes to use, although
                it can also be other tensors which extend AbstractTensor, such
                as custom tensors for Secure Multi-Party Computation or
                Federated Learning.
            location: An optional BaseWorker object which points to the worker
                on which this pointer's object can be found.
            id_at_location: An optional string or integer id of the tensor
                being pointed to.
            register: An optional boolean parameter to determine whether to
                automatically register the new pointer that gets created.
            owner: An optional BaseWorker object to specify the worker on which
                the pointer is located. It is also where the pointer is
                registered if register is set to True. Note that this is
                different from the location parameter that specifies where the
                pointer points to.
            id: An optional string or integer id of the PointerTensor.
            garbage_collect_data: If true (default), delete the remote tensor when the
                pointer is deleted.
            point_to_attr: string which can tell a pointer to not point directly to\
                a tensor, but to point to an attribute of that tensor (which must
                also be a tensor) such as .child or .grad. Note the string can be
                a chain (i.e., .child.child.child or .grad.child.child). Defaults
                to None, which meants don't point to any attr, just point to the
                tensor corresponding to the id_at_location.
        """
        super().__init__(tags, description)

        self.location = location
        self.id_at_location = id_at_location
        self.owner = owner
        self.id = id
        self.garbage_collect_data = garbage_collect_data
        self.shape = shape
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
        with the raising error RemoteTensorFoundError
        """
        try:
            cmd, _, args, kwargs = command
            _ = syft.frameworks.torch.hook_args.hook_function_args(cmd, args, kwargs)
        except RemoteTensorFoundError as err:
            pointer = err.pointer
            return pointer

    def __str__(self):
        """Returns a string version of this pointer.

        This is primarily for end users to quickly see things about the tensor.
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

        if self.tags is not None:
            big_str = True
            out += "\n\tTags: "
            for tag in self.tags:
                out += str(tag) + " "

        if big_str:
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

    def get(self, deregister_ptr: bool = True):
        """Requests the tensor/chain being pointed to, be serialized and return

        Since PointerTensor objects always point to a remote tensor (or chain
        of tensors, where a chain is simply a linked-list of tensors linked via
        their .child attributes), this method will request that the tensor/chain
        being pointed to be serialized and returned from this function.

        Note:
            This will typically mean that the remote object will be
            removed/destroyed. To just bring a copy back to the local worker,
            call .copy() before calling .get().


        Args:

            deregister_ptr (bool, optional): this determines whether to
                deregister this pointer from the pointer's owner during this
                method. This defaults to True because the main reason people use
                this method is to move the tensor from the remote machine to the
                local one, at which time the pointer has no use.

        Returns:
            An AbstractTensor object which is the tensor (or chain) that this
            object used to point to on a remote machine.

        TODO: add param get_copy which doesn't destroy remote if true.
        """

        if self.point_to_attr is not None:

            raise CannotRequestTensorAttribute(
                "You called .get() on a pointer to"
                " a tensor attribute. This is not yet"
                " supported. Call .clone().get() instead."
            )

        # if the pointer happens to be pointing to a local object,
        # just return that object (this is an edge case)
        if self.location == self.owner:
            tensor = self.owner.get_obj(self.id_at_location).child
        else:
            # get tensor from remote machine
            tensor = self.owner.request_obj(self.id_at_location, self.location)

        # Register the result
        assigned_id = self.id_at_location
        self.owner.register_obj(tensor, assigned_id)

        # Remove this pointer by default
        if deregister_ptr:
            self.owner.de_register_obj(self)

        # TODO: remove these 3 lines
        # The fact we have to check this means
        # something else is probably broken
        if tensor.is_wrapper:
            if isinstance(tensor.child, torch.Tensor):
                return tensor.child

        return tensor

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
                self.owner.send_msg(MSGTYPE.OBJ_DEL, self.id_at_location, self.location)

    @property
    def grad(self):
        if not hasattr(self, "_grad"):
            self._grad = self.attr("grad")

        if self._grad.child.is_none():
            return None

        return self._grad

    @grad.setter
    def grad(self, new_grad):
        self._grad = new_grad

    @property
    def data(self):
        if not hasattr(self, "_data"):
            self._data = self.attr("data")
        return self._data

    @data.setter
    def data(self, new_data):
        self._data = new_data

    def is_none(self):
        return self.owner.request_is_remote_tensor_none(self)

    def get_shape(self):

        results = self.location.search(str(self.id_at_location))

        if len(results) > 0:
            return results[0].shape
        else:
            print("couldn't find shape... are you sure this tensor exists?")

    @property
    def shape(self):
        """This method returns the shape of the data being pointed to.
        This shape information SHOULD be cached on self._shape, but
        occasionally this information may not be present. If this is the
        case, then it requests the shape information from the remote object
        directly (which is inefficient and should be avoided)."""

        if self._shape is None:
            self._shape = self.get_shape()

        return self._shape

    @shape.setter
    def shape(self, new_shape):
        self._shape = new_shape

    def attr(self, attr_name):
        if self.point_to_attr is not None:
            point_to_attr = "{}.{}".format(self.point_to_attr, attr_name)
        else:
            point_to_attr = attr_name

        attr_ptr = syft.PointerTensor(
            id=self.id,
            owner=self.owner,
            location=self.location,
            id_at_location=self.id_at_location,
            point_to_attr=point_to_attr,
        ).wrap()
        self.__setattr__(attr_name, attr_ptr)
        return attr_ptr
