import syft
import torch
from syft.frameworks.torch.tensors.interpreters import abstract
from syft.frameworks.torch import pointers

from syft.workers import AbstractWorker

from typing import List
from typing import Union


class PointerTensor(pointers.ObjectPointer, abstract.AbstractTensor):
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
    in the AbstractWorker object in self.location.

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
        location: "AbstractWorker" = None,
        id_at_location: Union[str, int] = None,
        owner: "AbstractWorker" = None,
        id: Union[str, int] = None,
        garbage_collect_data: bool = True,
        shape: torch.Size = None,
        point_to_attr: str = None,
        tags: List[str] = None,
        description: str = None,
    ):
        """Initializes a PointerTensor.

        Args:
            location: An optional AbstractWorker object which points to the worker
                on which this pointer's object can be found.
            id_at_location: An optional string or integer id of the object
                being pointed to.
            owner: An optional AbstractWorker object to specify the worker on which
                the pointer is located. It is also where the pointer is
                registered if register is set to True. Note that this is
                different from the location parameter that specifies where the
                pointer points to.
            id: An optional string or integer id of the PointerTensor.
            garbage_collect_data: If true (default), delete the remote object when the
                pointer is deleted.
            shape: size of the tensor the pointer points to
            point_to_attr: string which can tell a pointer to not point directly to\
                an object, but to point to an attribute of that object such as .child or
                .grad. Note the string can be a chain (i.e., .child.child.child or
                .grad.child.child). Defaults to None, which means don't point to any attr,
                just point to then object corresponding to the id_at_location.
            tags: an optional set of strings corresponding to this tensor
                which this tensor should be searchable for.
            description: an optional string describing the purpose of the tensor.
        """

        super().__init__(
            location=location,
            id_at_location=id_at_location,
            owner=owner,
            id=id,
            garbage_collect_data=garbage_collect_data,
            point_to_attr=point_to_attr,
            tags=tags,
            description=description,
        )
        self._shape = shape

    def get_shape(self):
        """Request information about the shape to the remote worker"""
        return self.owner.request_remote_tensor_shape(self)

    @property
    def shape(self):
        """ This method returns the shape of the data being pointed to.
        This shape information SHOULD be cached on self._shape, but
        occasionally this information may not be present. If this is the
        case, then it requests the shape information from the remote object
        directly (which is inefficient and should be avoided).
        """

        if self._shape is None:
            self._shape = self.get_shape()

        return self._shape

    @shape.setter
    def shape(self, new_shape):
        self._shape = new_shape

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

    @staticmethod
    def create_pointer(
        tensor,
        location: AbstractWorker = None,
        id_at_location: (str or int) = None,
        register: bool = False,
        owner: AbstractWorker = None,
        ptr_id: (str or int) = None,
        garbage_collect_data=None,
        shape=None,
        local_autograd=False,
        preinitialize_grad=False,
    ) -> "PointerTensor":
        """Creates a pointer to the "self" torch.Tensor object.

        This method is called on a torch.Tensor object, returning a pointer
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
            A torch.Tensor[PointerTensor] pointer to self. Note that this
            object will likely be wrapped by a torch.Tensor wrapper.
        """
        if owner is None:
            owner = tensor.owner

        if location is None:
            location = tensor.owner.id

        owner = tensor.owner.get_worker(owner)
        location = tensor.owner.get_worker(location)

        # previous_pointer = owner.get_pointer_to(location, id_at_location)
        previous_pointer = None

        if previous_pointer is None:
            ptr = PointerTensor(
                location=location,
                id_at_location=id_at_location,
                owner=owner,
                id=ptr_id,
                garbage_collect_data=True if garbage_collect_data is None else garbage_collect_data,
                shape=shape,
                tags=tensor.tags,
                description=tensor.description,
            )

        return ptr

    def move(self, location):
        ptr = self.owner.send(self, location)
        ptr.remote_get()
        # don't want it to accidentally delete the remote object
        # when this pointer is deleted
        ptr.garbage_collect_data = False
        return ptr

    def remote_get(self):
        self.owner.send_command(message=("mid_get", self, (), {}), recipient=self.location)
        return self

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
            object used to point to #on a remote machine.
        """
        tensor = pointers.ObjectPointer.get(self, deregister_ptr=deregister_ptr)

        # TODO: remove these 3 lines
        # The fact we have to check this means
        # something else is probably broken
        if tensor.is_wrapper:
            if isinstance(tensor.child, torch.Tensor):
                return tensor.child

        return tensor

    def attr(self, attr_name):
        attr_ptr = syft.PointerTensor(
            id=self.id,
            owner=self.owner,
            location=self.location,
            id_at_location=self.id_at_location,
            point_to_attr=self._create_attr_name_string(attr_name),
        ).wrap()
        self.__setattr__(attr_name, attr_ptr)
        return attr_ptr

    def dim(self) -> int:
        return len(self.shape)

    def fix_prec(self, *args, **kwargs):
        """
        Send a command to remote worker to transform a tensor to fix_precision

        Returns:
            A pointer to an FixPrecisionTensor
        """

        # Send the command
        command = ("fix_prec", self, args, kwargs)

        response = self.owner.send_command(self.location, command)

        return response

    fix_precision = fix_prec

    def share(self, *args, **kwargs):
        """
        Send a command to remote worker to additively share a tensor

        Returns:
            A pointer to an AdditiveSharingTensor
        """

        # Send the command
        command = ("share", self, args, kwargs)

        response = self.owner.send_command(self.location, command)

        return response

    def set_garbage_collect_data(self, value):
        self.garbage_collect_data = value

    def item(self) -> None:
        """
        Raising error with a message to be using .get instead of .item
        """
        raise RuntimeError(
            'Error, Please consider calling ".get" method instead of ".item" method, '
            "so you can be safely getting the item you need."
        )

    def __eq__(self, other):
        return self.eq(other)

    @staticmethod
    def simplify(ptr: "PointerTensor") -> tuple:
        """
        This function takes the attributes of a PointerTensor and saves them in a dictionary
        Args:
            ptr (pointers.PointerTensor): a PointerTensor
        Returns:
            tuple: a tuple holding the unique attributes of the pointer
        Examples:
            data = simplify(ptr)
        """

        return (
            ptr.id,
            ptr.id_at_location,
            ptr.location.id,
            ptr.point_to_attr,
            syft.serde._simplify(ptr._shape),
            ptr.garbage_collect_data,
        )

        # a more general but slower/more verbose option

        # data = vars(ptr).copy()
        # for k, v in data.items():
        #     if isinstance(v, AbstractWorker):
        #         data[k] = v.id
        # return _simplify_dictionary(data)

    @staticmethod
    def detail(worker: AbstractWorker, tensor_tuple: tuple) -> "PointerTensor":
        """
        This function reconstructs a PointerTensor given it's attributes in form of a dictionary.
        We use the spread operator to pass the dict data as arguments
        to the init method of PointerTensor
        Args:
            worker: the worker doing the deserialization
            tensor_tuple: a tuple holding the attributes of the PointerTensor
        Returns:
            PointerTensor: a pointers.PointerTensor
        Examples:
            ptr = detail(data)
        """
        # TODO: fix comment for this and simplifier
        obj_id, id_at_location, worker_id, point_to_attr, shape, garbage_collect_data = tensor_tuple

        if isinstance(worker_id, bytes):
            worker_id = worker_id.decode()

        if shape is not None:
            shape = torch.Size(syft.serde._detail(worker, shape))

        # If the pointer received is pointing at the current worker, we load the tensor instead
        if worker_id == worker.id:
            tensor = worker.get_obj(id_at_location)

            if point_to_attr is not None and tensor is not None:

                point_to_attrs = point_to_attr.decode("utf-8").split(".")
                for attr in point_to_attrs:
                    if len(attr) > 0:
                        tensor = getattr(tensor, attr)

                if tensor is not None:

                    if not tensor.is_wrapper and not isinstance(tensor, torch.Tensor):
                        # if the tensor is a wrapper then it doesn't need to be wrapped
                        # i the tensor isn't a wrapper, BUT it's just a plain torch tensor,
                        # then it doesn't need to be wrapped.
                        # if the tensor is not a wrapper BUT it's also not a torch tensor,
                        # then it needs to be wrapped or else it won't be able to be used
                        # by other interfaces
                        tensor = tensor.wrap()

            return tensor
        # Else we keep the same Pointer
        else:

            location = syft.torch.hook.local_worker.get_worker(worker_id)

            ptr = pointers.PointerTensor(
                location=location,
                id_at_location=id_at_location,
                owner=worker,
                id=obj_id,
                shape=shape,
                garbage_collect_data=garbage_collect_data,
            )

            return ptr

        # a more general but slower/more verbose option

        # new_data = {}
        # for k, v in data.items():
        #     key = k.decode()
        #     if type(v) is bytes:
        #         val_str = v.decode()
        #         val = syft.local_worker.get_worker(val_str)
        #     else:
        #         val = v
        #     new_data[key] = val
        # return PointerTensor(**new_data)
