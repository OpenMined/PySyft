from typing import List, Union

import syft
from syft.generic.frameworks.hook.hook_args import one
from syft.generic.frameworks.hook.hook_args import register_type_rule
from syft.generic.frameworks.hook.hook_args import register_forward_func
from syft.generic.frameworks.hook.hook_args import register_backward_func
from syft.generic.frameworks.types import FrameworkShapeType
from syft.generic.frameworks.types import FrameworkTensor
from syft.generic.abstract.tensor import AbstractTensor
from syft.generic.pointers.object_pointer import ObjectPointer
from syft.messaging.message import TensorCommandMessage
from syft.workers.abstract import AbstractWorker

from syft_proto.generic.pointers.v1.pointer_tensor_pb2 import PointerTensor as PointerTensorPB

from syft.exceptions import RemoteObjectFoundError

import torch


class PointerTensor(ObjectPointer, AbstractTensor):
    """A pointer to another tensor.

    A PointerTensor forwards all API calls to the remote tensor. PointerTensor
    objects point to tensors (as their name implies). They exist to mimic the
    entire API of a normal tensor, but instead of computing a tensor function
    locally (such as addition, subtraction, etc.) they forward the computation
    to a remote machine as specified by self.location. Specifically, every
    PointerTensor has a tensor located somewhere that it points to (they should
    never exist by themselves). Note that PointerTensor objects can point to
    both FrameworkTensor objects AND to other PointerTensor objects. Furthermore,
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
        shape: FrameworkShapeType = None,
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
        """This method returns the shape of the data being pointed to.
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
        try:
            return self.owner.request_is_remote_tensor_none(self)
        except:
            """TODO: this might hide useful errors, but we don't have good
            enough remote error handling yet to do anything better."""
            return True

    def clone(self, memory_format=torch.preserve_format):
        """
        Clone should keep ids unchanged, contrary to copy.
        We make the choice that a clone action is local, and can't affect
        the remote tensors, so garbage_collect_data is always False, both
        for the tensor cloned and the clone.
        """
        self.garbage_collect_data = False
        cloned_tensor = type(self)(**self.get_class_attributes())
        cloned_tensor.id = self.id
        cloned_tensor.owner = self.owner

        return cloned_tensor

    def get_class_attributes(self):
        """
        Used for cloning (see AbtractTensor)
        """
        return {
            "location": self.location,
            "id_at_location": self.id_at_location,
            "garbage_collect_data": self.garbage_collect_data,
        }

    @staticmethod
    def create_pointer(
        tensor,
        location: Union[AbstractWorker, str] = None,
        id_at_location: (str or int) = None,
        owner: Union[AbstractWorker, str] = None,
        ptr_id: (str or int) = None,
        garbage_collect_data=None,
        shape=None,
    ) -> "PointerTensor":
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
            owner: A AbstractWorker parameter to specify the worker on which the
                pointer is located. It is also where the pointer is registered
                if register is set to True.
            ptr_id: A string or integer parameter to specify the id of the pointer
                in case you wish to set it manually for any special reason.
                Otherwise, it will be set randomly.
            garbage_collect_data: If true (default), delete the remote tensor when the
                pointer is deleted.

        Returns:
            A FrameworkTensor[PointerTensor] pointer to self. Note that this
            object itself will likely be wrapped by a FrameworkTensor wrapper.
        """
        if owner is None:
            owner = tensor.owner

        if location is None:
            location = tensor.owner

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

    def move(self, destination: AbstractWorker, requires_grad: bool = False):
        """
        Will move the remove value from self.location A to destination B
        Note a A will keep a copy of his value that he sent to B. This follows the
        .send() paradigm where the local worker keeps a copy of the value he sends.
        Args:
            destination: the new location of the remote data
            requires_grad: see send() for details
        Returns:
            A pointer to location
        """
        # move to local target is equivalent to doing .get()
        if self.owner.id == destination.id:
            return self.get()

        if self.location.id == destination.id:
            return self

        ptr = self.remote_send(destination, requires_grad=requires_grad)

        # We make the pointer point at the remote value. As the id doesn't change,
        # we don't update ptr.id_at_location. See issue #3217 about this.
        # Note that you have now 2 pointers on different locations pointing to the
        # same tensor.
        ptr.location = destination

        return ptr

    def remote_send(self, destination: AbstractWorker, requires_grad: bool = False):
        """Request the worker where the tensor being pointed to belongs to send it to destination.
        For instance, if C holds a pointer, ptr, to a tensor on A and calls ptr.remote_send(B),
        C will hold a pointer to a pointer on A which points to the tensor on B.
        Args:
            destination: where the remote value should be sent
            requires_grad: if true updating the grad of the remote tensor on destination B will
                trigger a message to update the gradient of the value on A.
        """
        kwargs_ = {"inplace": False, "requires_grad": requires_grad}
        message = TensorCommandMessage.communication(
            "remote_send", self, (destination.id,), kwargs_, (self.id,)
        )
        self.owner.send_msg(message=message, location=self.location)
        return self

    def remote_get(self):
        self.owner.send_command(cmd_name="mid_get", target=self, recipient=self.location)
        return self

    def get(self, user=None, reason: str = "", deregister_ptr: bool = True):
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
            user (obj, optional): user credentials to perform authentication process.
            reason (str, optional): a description of why the data scientist wants to see it.
            deregister_ptr (bool, optional): this determines whether to
                deregister this pointer from the pointer's owner during this
                method. This defaults to True because the main reason people use
                this method is to move the tensor from the remote machine to the
                local one, at which time the pointer has no use.

        Returns:
            An AbstractTensor object which is the tensor (or chain) that this
            object used to point to #on a remote machine.
        """
        tensor = ObjectPointer.get(self, user=user, reason=reason, deregister_ptr=deregister_ptr)

        # TODO: remove these 3 lines
        # The fact we have to check this means
        # something else is probably broken
        if tensor.is_wrapper:
            if isinstance(tensor.child, FrameworkTensor):
                return tensor.child
        return tensor

    def attr(self, attr_name):
        attr_ptr = PointerTensor(
            id=self.id,
            owner=self.owner,
            location=self.location,
            id_at_location=self.id_at_location,
            point_to_attr=self._create_attr_name_string(attr_name),
        ).wrap(register=False)
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
        response = self.owner.send_command(self.location, "fix_prec", self, args, kwargs)
        return response

    fix_precision = fix_prec

    def float_prec(self, *args, **kwargs):
        """
        Send a command to remote worker to transform a fix_precision tensor back to float_precision

        Returns:
            A pointer to a Tensor
        """
        response = self.owner.send_command(self.location, "float_prec", self, args, kwargs)
        return response

    float_precision = float_prec

    def share(self, *args, **kwargs):
        """
        Send a command to remote worker to additively share a tensor

        Returns:
            A pointer to an AdditiveSharingTensor
        """
        if len(args) < 2:
            raise RuntimeError("Error, share must have > 1 arguments all of type syft.workers")

        response = self.owner.send_command(self.location, "share", self, args, kwargs)
        return response

    def share_(self, *args, **kwargs):
        """
        Send a command to remote worker to additively share inplace a tensor

        Returns:
            A pointer to an AdditiveSharingTensor
        """
        response = self.owner.send_command(self.location, "share_", self, args, kwargs)
        return self

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

    def __iter__(self):
        return (self[idx] for idx in range(self.shape[0]))

    @staticmethod
    def simplify(worker: AbstractWorker, ptr: "PointerTensor") -> tuple:
        """
        This function takes the attributes of a PointerTensor and saves them in a dictionary
        Args:
            worker (AbstractWorker): the worker doing the serialization
            ptr (PointerTensor): a PointerTensor
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
            syft.serde.msgpack.serde._simplify(worker, ptr._shape),
            ptr.garbage_collect_data,
            syft.serde.msgpack.serde._simplify(worker, ptr.tags),
            ptr.description,
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
            PointerTensor: a PointerTensor
        Examples:
            ptr = detail(data)
        """
        # TODO: fix comment for this and simplifier
        (
            obj_id,
            id_at_location,
            worker_id,
            point_to_attr,
            shape,
            garbage_collect_data,
            tags,
            description,
        ) = tensor_tuple

        obj_id = syft.serde.msgpack.serde._detail(worker, obj_id)
        id_at_location = syft.serde.msgpack.serde._detail(worker, id_at_location)
        worker_id = syft.serde.msgpack.serde._detail(worker, worker_id)
        point_to_attr = syft.serde.msgpack.serde._detail(worker, point_to_attr)

        if shape is not None:
            shape = syft.hook.create_shape(syft.serde.msgpack.serde._detail(worker, shape))

        # If the pointer received is pointing at the current worker, we load the tensor instead
        if worker_id == worker.id:
            tensor = worker.get_obj(id_at_location)

            if point_to_attr is not None and tensor is not None:

                point_to_attrs = point_to_attr.split(".")
                for attr in point_to_attrs:
                    if len(attr) > 0:
                        tensor = getattr(tensor, attr)

                if tensor is not None:

                    if not tensor.is_wrapper and not isinstance(tensor, FrameworkTensor):
                        # if the tensor is a wrapper then it doesn't need to be wrapped
                        # if the tensor isn't a wrapper, BUT it's just a plain torch tensor,
                        # then it doesn't need to be wrapped.
                        # if the tensor is not a wrapper BUT it's also not a torch tensor,
                        # then it needs to be wrapped or else it won't be able to be used
                        # by other interfaces
                        tensor = tensor.wrap()

            return tensor
        # Else we keep the same Pointer
        else:

            location = syft.hook.local_worker.get_worker(worker_id)

            ptr = PointerTensor(
                location=location,
                id_at_location=id_at_location,
                owner=worker,
                id=obj_id,
                shape=shape,
                garbage_collect_data=garbage_collect_data,
                tags=tags,
                description=description,
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

    @staticmethod
    def bufferize(worker: AbstractWorker, ptr: "PointerTensor") -> PointerTensorPB:
        protobuf_pointer = PointerTensorPB()

        syft.serde.protobuf.proto.set_protobuf_id(protobuf_pointer.object_id, ptr.id)
        syft.serde.protobuf.proto.set_protobuf_id(protobuf_pointer.location_id, ptr.location.id)
        syft.serde.protobuf.proto.set_protobuf_id(
            protobuf_pointer.object_id_at_location, ptr.id_at_location
        )

        if ptr.point_to_attr:
            protobuf_pointer.point_to_attr = ptr.point_to_attr
        protobuf_pointer.garbage_collect_data = ptr.garbage_collect_data
        return protobuf_pointer

    @staticmethod
    def unbufferize(worker: AbstractWorker, protobuf_tensor: PointerTensorPB) -> "PointerTensor":
        # Extract the field values

        obj_id = syft.serde.protobuf.proto.get_protobuf_id(protobuf_tensor.object_id)
        obj_id_at_location = syft.serde.protobuf.proto.get_protobuf_id(
            protobuf_tensor.object_id_at_location
        )
        worker_id = syft.serde.protobuf.proto.get_protobuf_id(protobuf_tensor.location_id)
        point_to_attr = protobuf_tensor.point_to_attr
        shape = syft.hook.create_shape(protobuf_tensor.shape.dims)
        garbage_collect_data = protobuf_tensor.garbage_collect_data

        # If the pointer received is pointing at the current worker, we load the tensor instead
        if worker_id == worker.id:
            tensor = worker.get_obj(obj_id_at_location)

            if point_to_attr is not None and tensor is not None:

                point_to_attrs = point_to_attr.split(".")
                for attr in point_to_attrs:
                    if len(attr) > 0:
                        tensor = getattr(tensor, attr)

                if tensor is not None:

                    if not tensor.is_wrapper and not isinstance(tensor, FrameworkTensor):
                        # if the tensor is a wrapper then it doesn't need to be wrapped
                        # if the tensor isn't a wrapper, BUT it's just a plain torch tensor,
                        # then it doesn't need to be wrapped.
                        # if the tensor is not a wrapper BUT it's also not a torch tensor,
                        # then it needs to be wrapped or else it won't be able to be used
                        # by other interfaces
                        tensor = tensor.wrap()

            return tensor
        # Else we keep the same Pointer
        else:
            location = syft.hook.local_worker.get_worker(worker_id)

            ptr = PointerTensor(
                location=location,
                id_at_location=obj_id_at_location,
                owner=worker,
                id=obj_id,
                shape=shape,
                garbage_collect_data=garbage_collect_data,
            )

            return ptr

    @staticmethod
    def get_protobuf_schema() -> PointerTensorPB:
        return PointerTensorPB


### Register the tensor with hook_args.py ###
register_type_rule({PointerTensor: one})
register_forward_func({PointerTensor: lambda p: (_ for _ in ()).throw(RemoteObjectFoundError(p))})
register_backward_func({PointerTensor: lambda i: i})
