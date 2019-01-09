from syft.frameworks.torch.tensors.abstract import AbstractTensor
from syft.codes import MSGTYPE


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
        """

        self.location = location
        self.id_at_location = id_at_location
        self.owner = owner
        self.id = id
        self.garbage_collect_data = garbage_collect_data

    def __str__(self):
        """Returns a string version of this pointer.

        This is primarily for end users to quickly see things about the tensor.
        This tostring shouldn't be used for anything else though as it's likely
        to change. (aka, don't try to parse it to extract information. Read the
        attribute you need directly). Also, don't use this to-string as a
        serialized form of the pointer.
        """

        type_name = type(self).__name__
        return (
            f"["
            f"{type_name} - "
            f"id:{self.id} "
            f"owner:{self.owner.id} "
            f"loc:{self.location.id} "
            f"id@loc:{self.id_at_location}"
            f"]"
        )

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

            out (class:`.abstract.AbstractTensor`): this is the tensor (or
                chain) which this object used to point to on a remote machine.

        Returns:
            tensor:

        #TODO: add param get_copy which doesn't destroy remote if true.

        Since PointerTensor objects always point to a remote tensor (or chain
        of tensors, where a chain is simply a linked-list of tensors linked via
        their .child attributes), this method will request that the tensor or
        chain being pointed to be serialized and returned from this function.
        This will typically mean that the remote object will be removed or
        destroyed. If you merely wish to bring a copy back to the local worker,
        call .copy() before calling .get().
        TODO: add param get_copy which doesn't destroy remote if true.

        Args:
            deregister_ptr: An optional boolean parameter (default True) that
                determines whether to deregister this pointer from the
                pointer's owner during this method. The default is set to True
                because the main reason people use this method is to move the
                tensor from the remote machine to the local one, at which time
                the pointer has no use.

        Returns:
            An AbstractTensor object which is the tensor (or chain) that this
            object used to point to on a remote machine.
        """

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
            self.owner.send_msg(MSGTYPE.OBJ_DEL, self.id_at_location, self.location)
