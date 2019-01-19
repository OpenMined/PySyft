import random
import weakref

import syft
from syft.frameworks.torch.tensors import AbstractTensor
from syft.frameworks.torch.tensors import PointerTensor
from syft.workers import BaseWorker

from syft.exceptions import PureTorchTensorFoundError


class TorchTensor(AbstractTensor):
    """Add methods to this tensor to have them added to every torch.Tensor object.

    This tensor is simply a more convenient way to add custom functions to
    all Torch tensor types. When you add a function to this tensor, it will
    be added to EVERY native torch tensor type (i.e. torch.Torch) automatically
    by the TorchHook (which is in frameworks/torch/hook.py).

    Note: all methods from AbstractTensor will also be included because this
    tensor extends AbstractTensor. So, if you're looking for a method on
    the native torch tensor API but it's not listed here, you might try
    checking AbstractTensor.
    """

    def __str__(self) -> str:
        if hasattr(self, "child"):
            return type(self).__name__ + ">" + self.child.__str__()
        else:
            return self.native___str__()

    def __repr__(self) -> str:
        if hasattr(self, "child"):
            return type(self).__name__ + ">" + self.child.__repr__()
        else:
            return self.native___repr__()

    @classmethod
    def handle_func_command(cls, command):
        """
        Receive an instruction for a function to be applied on a torch
        tensor, which can be a "real" tensor or just a wrapper at the
        top of a chain (ex: wrapper>LoggingTensor>Torch tensor).
        If this is not a wrapper layer, run the native torch command.
        If this is a wrapper layer, just forward the instruction to the
        next layer type in the chain (in the example above to LoggingTensor.
        handle_method_command), get the response and replace a wrapper
        on top of all tensors found in the response.
        :param command: instruction of a function command: (command name,
        <no self>, arguments[, kwargs])
        :return: the response of the function command
        """
        # TODO: add kwargs
        cmd, _, args = command

        try:  # will work if tensors are wrappers
            # Replace all torch tensor with their child attribute
            new_args, new_type = syft.frameworks.torch.hook_args.hook_function_args(cmd, args)
            # build the new command
            new_command = (cmd, None, new_args)
            # Send it to the appropriate class and get the response
            response = new_type.handle_func_command(new_command)
            # Put back the wrappers where needed
            response = syft.frameworks.torch.hook_args.hook_response(cmd, response, wrap_type=cls)
        except PureTorchTensorFoundError:  # means that it's not a wrapper but a pure tensor
            # TODO: clean this line
            cmd = (
                "syft.local_worker.hook."
                + ".".join(cmd.split(".")[:-1])
                + ".native_"
                + cmd.split(".")[-1]
            )
            # Run the native function with the new args
            # Note the the cmd should already be checked upon reception by the worker
            # in the execute_command function
            if isinstance(args, tuple):
                response = eval(cmd)(*args)
            else:
                response = eval(cmd)(args)

        return response

    def send(self, location):
        """Gets the pointer to a new remote object.

        One of the most commonly used methods in PySyft, this method serializes
        the object upon which it is called (self), sends the object to a remote
        worker, creates a pointer to that worker, and then returns that pointer
        from this function.

        Args:
            location: The BaseWorker object which you want to send this object
                to. Note that this is never actually the BaseWorker but instead
                a class which instantiates the BaseWorker abstraction.

        Returns:
            A torch.Tensor[PointerTensor] pointer to self. Note that this
            object will likely be wrapped by a torch.Tensor wrapper.
        """

        # If you send a pointer p1, you want the pointer to pointer p2 to control
        # the garbage collection and not the remaining old p1 (here self). Because if
        # p2 is not GCed, GCing p1 shouldn't delete the remote tensor, but if you
        # want to do so, as p2 is not GCed, you can still do `del p2`.
        # This allows to chain multiple .send().send() calls.
        if hasattr(self, "child") and isinstance(self.child, PointerTensor):
            self.child.garbage_collect_data = False

        ptr = self.owner.send(self, location)

        # The last pointer should control remote GC, not the previous self.ptr
        if hasattr(self, "ptr"):
            self.ptr().garbage_collect_data = False

        # we need to cache this weak reference to the pointer so that
        # if this method gets called multiple times we can simply re-use
        # the same pointer which was previously created
        self.ptr = weakref.ref(ptr)

        return ptr.wrap()

    def create_pointer(
        self,
        location: BaseWorker = None,
        id_at_location: (str or int) = None,
        register: bool = False,
        owner: BaseWorker = None,
        ptr_id: (str or int) = None,
        garbage_collect_data: bool = True,
    ) -> PointerTensor:
        """Creates a pointer to the "self" torch.Tensor object.

        This method is called on a torch.Tensor object, returning a pointer
        to that object. This method is the CORRECT way to create a pointer,
        and the parameters of this method give all possible attributes that
        a pointer can be created with.

        Args:
            location: The BaseWorker object which points to the worker on which
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
            owner: A BaseWorker parameter to specify the worker on which the
                pointer is located. It is also where the pointer is registered
                if register is set to True.
            ptr_id: A string or integer parameter to specify the id of the pointer
                in case you wish to set it manually for any special reason.
                Otherwise, it will be set randomly.
            garbage_collect_data: If true (default), delete the remote tensor when the
                pointer is deleted.

        Returns:
            A torch.Tensor[PointerTensor] pointer to self. Note that this
            object will likely be wrapped by a torch.Tensor wrapper.
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
                garbage_collect_data=garbage_collect_data,
            )

        return ptr

    def get(self, deregister_ptr: bool = True):
        """Requests the tensor/chain being pointed to, be serialized and return
        """
        # Transfer the get() to the child attribute which is a pointer
        tensor = self.child.get()

        # Clean the wrapper
        delattr(self, "child")

        return tensor
