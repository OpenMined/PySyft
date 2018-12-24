import random

from . import AbstractTensor, PointerTensor
from ....workers import BaseWorker



class TorchTensor(AbstractTensor):
    """ Add methods to this tensor to have them added to every torch.Tensor obj


    This tensor is simply a more convenient way to add custom functions to
    all Torch tensor types. When you add a function to this tensor, it will
    be added to EVERY native torch tensor type (i.e. torch.Torch) automatically
    by the TorchHook (which is in frameworks/torch/hook.py)

    Note: all methods from AbstractTensor will also be included because this
    tensor extends AbstractTensor. So, if you're looking for a method on
    the native torch tensor API but it's not listed here, you might try
    checking AbstractTensor.
    """

    def get(self):
        """Get remote data from remove places and send it here to the client.

        Many tensor types point to data which exists at another location, including
        PointerTensor and various Secure Multi-Party Computation tensors. All of these
        tensor types have a .get() method which requests that the data on the remote
        machine be set back to the client (to the machine upon which .get() is called).

        Returns:
            AbstractTensor: Typically, this method will return a single tensor, although
                that tensor might be a combination of several remote tensors which
                exist on several different machines. If the method you're trying to
                call/implement needs to return multiple tensors or has some very special
                way in which those tensors should be combined, consider creating a
                special method specifically for that functionality.

        """
        return self.child.get()

    def send(self, location):
        """Send self to a remote worker and return a pointer to the new remote object

        One of the most commonly used methods in PySyft, this method serializes the
        object upon which it is called (self), sends the object to a remote worker,
        creates a pointer to that worker, and then returns that pointer from this
        function.

        Args:
            location (....workers.BaseWorker): the BaseWorker object which you want
                to send this object to. Note that this is never actually
                BaseWorker but instead a class which instantiates the BaseWorker
                abstraction.

        Returns:
            torch.Tensor[PointerTensor]: This method returns the pointer to self.
                Note that this object will likely be wrapped by a torch.Tensor
                wrapper.

        """

        return self.owner.send(self, location)

    def create_pointer(
        self,
        location: BaseWorker = None,
        id_at_location: (str or int) = None,
        register: bool = False,
        owner: BaseWorker = None,
        ptr_id: (str or int) = None,
    ) -> PointerTensor:
        """This method creates a pointer to the "self" torch.Tensor object

        This method is called on a torch.Tensor object, returning a pointer
        to that object. This method is the CORRECT way to create a pointer, and
        the parameters of this method give all possible attributes that a pointer
        can be created with.

        Args:
            location (....workers.BaseWorker): the BaseWorker object which points
                to the worker on which this pointer's object can be found. In nearly
                all cases, this is self.owner and so this attribute can usually be
                left blank. Very rarely you may know that you're about to move the
                Tensor to another worker so you can pre-initialize the .location
                attribute of the pointer to some other worker, but this is a rare
                exception.

            id_at_location (str or int): the id of the tensor being pointed to.
                Similar to location, this parameter is almost always self.id and so
                you can leave this parameter to None. The only exception is if you
                happen to know that the ID is going to be something different than
                self.id, but again this is very rare and most of the time, setting
                this means that you're probably doing something you shouldn't.

            register (bool): this parameter determines whether to
                register the new pointer that gets created. By default, this is set
                to false because (most of the time) a pointer is initialized in this
                way so that it can be sent to someone else. (i.e., "oh you need to
                point to my tensor? let me create a pointer and send it to you").
                Thus, when a pointer gets created, we want to skip being registered
                on the local worker because the pointer is about to be sent
                elsewhere. However, if you are initializing a pointer you intend
                to keep, then it's probably a good idea to register it, especially
                if there's any chance that someone else will initialize a pointer to
                your pointer.

            owner (....workers.BaseWorker): while "location" specifies where the
                pointer points to, this parameter specifies the worker on which the
                pointer is located. It is also where the pointer is registered if
                register is set to True.

            id (str or int): if you want to set the id of the pointer for
                any special reason, you can set it here. Otherwise, it will be set
                randomly.

        Returns:
            torch.Tensor[PointerTensor]: This method returns the pointer to self.
                Note that this object will likely be wrapped by a torch.Tensor
                wrapper.

        """

        if owner is None:
            owner = self.owner

        if location is None:
            location = self.owner.id

        owner = self.owner.get_worker(owner)
        location = self.owner.get_worker(location)

        if id_at_location is None:
            id_at_location = self.id

        if ptr_id is None:
            if location.id != self.owner.id:
                ptr_id = self.id
            else:
                ptr_id = int(10e10 * random.random())

        # previous_pointer = owner.get_pointer_to(location, id_at_location)
        previous_pointer = None

        if previous_pointer is None:
            ptr = PointerTensor(
                parent=self,
                location=location,
                id_at_location=id_at_location,
                register=register,
                owner=owner,
                id=ptr_id,
            )

        return ptr.wrap()

    def reshape(self, *args, **kwargs):
        """This method reshapes a tensor to have new dimensions

        This method is the same functionality as the default reshape function which
        ships with PyTorch. See the PyTorch documentation for the correct args and
        kwargs. TODO: add link to documentation (i'm on a plane and don't have wifi)
        """

        return getattr(self, "native_reshape")(*args, **kwargs)
