import weakref

import syft as sy

# TODO: Remove this import, since generic classes shouldn't depend on torch
import torch

from syft.generic.abstract.object import AbstractObject
from syft.generic.pointers.pointer_tensor import PointerTensor
from syft.serde.syft_serializable import SyftSerializable
from syft.workers.abstract import AbstractWorker

from syft.exceptions import InvalidTensorForRemoteGet
from syft.exceptions import SendNotPermittedError


class AbstractSendable(AbstractObject, SyftSerializable):
    """
    This layers functionality for sending objects between workers on top of AbstractObject.
    """

    def serialize(self):  # check serde.py to see how to provide compression schemes
        """Serializes the tensor on which it's called.

        This is the high level convenience function for serializing torch
        tensors. It includes three steps, Simplify, Serialize, and Compress as
        described in serde.py.
        By default serde is compressing using LZ4

        Returns:
            The serialized form of the tensor.
            For example:
                x = torch.Tensor([1,2,3,4,5])
                x.serialize() # returns a serialized object
        """
        return sy.serde.serialize(self)

    def ser(self, *args, **kwargs):
        return self.serialize(*args, **kwargs)

    def get(self):
        """Just a pass through. This is most commonly used when calling .get() on a
        Syft tensor which has a child which is a pointer, an additive shared tensor,
        a multi-pointer, etc."""
        class_attributes = self.get_class_attributes()
        return type(self)(
            **class_attributes,
            owner=self.owner,
            tags=self.tags,
            description=self.description,
            id=self.id,
        ).on(self.child.get())

    def mid_get(self):
        """This method calls .get() on a child pointer and correctly registers the results"""

        child_id = self.id
        tensor = self.get()
        tensor.id = child_id
        self.owner.register_obj(tensor)

    # Communication methods
    def send(
        self,
        *location,
        inplace: bool = False,
        user: object = None,
        local_autograd: bool = False,
        requires_grad: bool = False,
        preinitialize_grad: bool = False,
        no_wrap: bool = False,
        garbage_collect_data: bool = True,
    ):
        """Gets the pointer to a new remote object.

        One of the most commonly used methods in PySyft, this method serializes the object upon
        which it is called (self), sends the object to a remote worker, creates a pointer to
        that worker, and then returns that pointer from this function.

        Args:
            location: The BaseWorker object which you want to send this object to. Note that
                this is never actually the BaseWorker but instead a class which instantiates the
                BaseWorker abstraction.
            inplace: if true, return the same object instance, else a new wrapper
            user (object,optional): User credentials to be verified.
            local_autograd: Use autograd system on the local machine instead of PyTorch's
                autograd on the workers.
            requires_grad: Default to False. If true, whenever the remote value of this tensor
                will have its gradient updated (for example when calling .backward()), a call
                will be made to set back the local gradient value.
            preinitialize_grad: Initialize gradient for AutogradTensors to a tensor
            no_wrap: If True, wrap() is called on the created pointer
            garbage_collect_data: argument passed down to create_pointer()

        Returns:
            A torch.Tensor[PointerTensor] pointer to self. Note that this
            object will likely be wrapped by a torch.Tensor wrapper.

        Raises:
                SendNotPermittedError: Raised if send is not permitted on this tensor.
        """

        if not self.allow(user=user):
            raise SendNotPermittedError()

        # If you send a pointer p1, you want the pointer to pointer p2 to control
        # the garbage collection and not the remaining old p1 (here self). Because if
        # p2 is not GCed, GCing p1 shouldn't delete the remote tensor, but if you
        # want to do so, as p2 is not GCed, you can still do `del p2`.
        # This allows to chain multiple .send().send() calls.

        if len(location) == 1:

            location = location[0]

            if hasattr(self, "child") and isinstance(self.child, PointerTensor):
                self.child.garbage_collect_data = False
                if self._is_parameter():
                    self.data.child.garbage_collect_data = False

            ptr = self.owner.send(
                self,
                location,
                local_autograd=local_autograd,
                requires_grad=requires_grad,
                preinitialize_grad=preinitialize_grad,
                garbage_collect_data=garbage_collect_data,
            )

            ptr.description = self.description
            ptr.tags = self.tags

            # The last pointer should control remote GC, not the previous self.ptr
            if hasattr(self, "ptr") and self.ptr is not None:
                ptr_ = self.ptr()
                if ptr_ is not None:
                    ptr_.garbage_collect_data = False

            # we need to cache this weak reference to the pointer so that
            # if this method gets called multiple times we can simply re-use
            # the same pointer which was previously created
            self.ptr = weakref.ref(ptr)

            if self._is_parameter():
                if inplace:
                    self.is_wrapper = True
                    with torch.no_grad():
                        self.set_()
                    self.data = ptr
                    output = self
                else:
                    if no_wrap:
                        raise ValueError("Parameters can't accept no_wrap=True")
                    wrapper = torch.Tensor()
                    param_wrapper = torch.nn.Parameter(wrapper)
                    param_wrapper.is_wrapper = True
                    with torch.no_grad():
                        param_wrapper.set_()
                    param_wrapper.data = ptr
                    output = param_wrapper
            else:
                if inplace:
                    self.is_wrapper = True
                    self.set_()
                    self.child = ptr
                    return self
                else:
                    output = ptr if no_wrap else ptr.wrap()

            if self.requires_grad:
                # This is for AutogradTensor to work on MultiPointerTensors
                # With pre-initialized gradients, this should get it from AutogradTensor.grad
                if preinitialize_grad:
                    grad = output.child.grad
                else:
                    grad = output.attr("grad")

                output.grad = grad

                # Because of the way PyTorch works, .grad is prone to
                # create entirely new Python objects for the tensor, which
                # inadvertently deletes our custom attributes (like .child)
                # But, if we keep a backup reference around, PyTorch seems
                # to re-use it, which means .grad keeps the attributes we
                # want it to keep. #HackAlert
                output.backup_grad = grad

            if local_autograd:
                output = sy.AutogradTensor(data=output, preinitialize_grad=preinitialize_grad).on(
                    output
                )

        else:

            children = []
            for loc in location:
                children.append(self.clone().send(loc, no_wrap=True))

            output = sy.MultiPointerTensor(children=children)

            if not no_wrap:
                output = output.wrap()

        return output

    def send_(self, *location, **kwargs):
        """
        Calls send() with inplace option, but only with a single location
        :param location: workers locations
        :return:
        """
        if len(location) > 1:
            raise NotImplementedError("Inplace send to several workers is currently not supported.")

        return self.send(*location, inplace=True, **kwargs)

    def create_pointer(
        self,
        location: AbstractWorker = None,
        id_at_location: (str or int) = None,
        owner: AbstractWorker = None,
        ptr_id: (str or int) = None,
        garbage_collect_data: bool = True,
        shape=None,
        **kwargs,
    ) -> PointerTensor:
        """Creates a pointer to the "self" torch.Tensor object.

        Returns:
            A PointerTensor pointer to self. Note that this
            object will likely be wrapped by a torch.Tensor wrapper.
        """
        if id_at_location is None:
            id_at_location = self.id

        if ptr_id is None:
            if location is not None and location.id != self.owner.id:
                ptr_id = self.id
            else:
                ptr_id = sy.ID_PROVIDER.pop()

        if shape is None:
            shape = self.shape

        ptr = sy.PointerTensor.create_pointer(
            self, location, id_at_location, owner, ptr_id, garbage_collect_data, shape
        )

        return ptr

    def mid_get(self):
        """This method calls .get() on a child pointer and correctly registers the results"""
        if not hasattr(self, "child"):
            raise InvalidTensorForRemoteGet(self)

        self.child.mid_get()

    def remote_get(self):
        """Assuming .child is a PointerTensor, this method calls .get() on the tensor
        that the .child is pointing to (which should also be a PointerTensor)

        TODO: make this kind of message forwarding generic?
        """
        if not hasattr(self, "child"):
            raise InvalidTensorForRemoteGet(self)

        self.child.remote_get()

        return self

    def get(self, *args, inplace: bool = False, user=None, reason: str = "", **kwargs):
        """Requests the tensor/chain being pointed to, be serialized and return
        Args:
            args: args to forward to worker
            inplace: if true, return the same object instance, else a new wrapper
            kwargs: kwargs to forward to worker
        Raises:
            GetNotPermittedError: Raised if get is not permitted on this tensor
        """

        # If it is a local tensor/chain, we don't need to verify permissions
        if not isinstance(self.child, sy.PointerTensor):
            tensor = self.child.get(*args, **kwargs)
        else:  # Remote tensor/chain
            tensor = self.child.get(*args, user=user, reason=reason, **kwargs)

        # Clean the wrapper
        delattr(self, "child")

        # Parameters use .data instead of children
        # so we need to have special support to make sure
        # that Parmeters operate inline (because they're
        # typically being managed inside of a model/optimizer
        # so not using the same wrapper can cause the model/
        # optimizer to lose track of where the actual weights
        # are.
        if isinstance(self, torch.nn.Parameter):
            self.is_wrapper = tensor.data.is_wrapper
            if inplace:
                self.data = tensor.data
                self.grad = tensor.grad
                return self
            else:
                return tensor

        if inplace:
            self.set_(tensor)
            if hasattr(tensor, "child"):
                self.child = tensor.child
            else:
                self.is_wrapper = False
            return self
        else:
            return tensor

    def get_(self, *args, **kwargs):
        """
        Calls get() with inplace option set to True
        """
        return self.get(*args, inplace=True, **kwargs)

    def allow(self, user=None) -> bool:
        """ This function returns will return True if it isn't a PrivateTensor, otherwise it will
        return the result of PrivateTensor's allow method.

            Args:
                user (object,optional): User credentials to be verified.

            Returns:
                boolean: If it is a public tensor/ allowed user, returns true, otherwise it returns
                false.
        """
        # If it is a wrapper
        if self.is_wrapper:
            current_tensor = self.child

            # Verify permissions for each element on the tensor chain.
            while hasattr(current_tensor, "child"):

                # If it has a list of allowed users, verify permissions,
                # otherwise (public tensors) go to the next.
                if hasattr(current_tensor, "allowed_users"):
                    allow = current_tensor.allow(user)
                    if not allow:
                        return False

                # Go to next element on the tensor chain
                current_tensor = current_tensor.child
        return True

    def move(self, location: AbstractWorker, requires_grad: bool = False):
        """
        Move acts on a pointer to A to move the remote value to B (=location).

        Note a A will keep a copy of his value that he sent to B. This follows the
        .send() paradigm where the local worker keeps a copy of the value he sends.

        Args:
            location: the worker where the remote value should be moved
            requires_grad: see send() for details

        Returns:
            A pointer to the worker location
        """
        new_ptr = self.child.move(location, requires_grad)
        # We get the owner from self.child because the owner of a wrapper is
        # not reliable and sometimes end up being the sy.local_worker
        self.child.owner.register_obj(self)
        if isinstance(new_ptr, PointerTensor):
            return new_ptr.wrap()
        else:
            return new_ptr

    def move_(self, location: AbstractWorker, requires_grad: bool = False):
        """
        Inplace version of move
        """
        new_ptr = self.move(location, requires_grad)
        self.child = new_ptr
        return self

    def remote_send(self, location):
        return self.child.remote_send(location).wrap()
