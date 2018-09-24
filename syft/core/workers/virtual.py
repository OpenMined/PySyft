from .base import BaseWorker


class VirtualWorker(BaseWorker):
    r"""
    A virtualized worker simulating the existence of a remote machine.
    It is intended as a testing, development, and performance evaluation
    tool that exists independent of a live or local network of workers.
    You don't even have to be connected to the internet to create a pool
    of Virtual Workers and start running computations on them.

    :Parameters:


    * **hook (**:class:`.hooks.BaseHook` **)** This is a reference to
      the hook object which overloaded the underlying deep learning framework.

    * **id (int or string, optional)** the integer or string identifier
      for this node

    * **is_client_worker (bool, optional)** a boolean which determines
      whether this worker is associeted with an end user client. If so,
      it assumes that the client will maintain control over when
      tensors/variables/models are instantiated or deleted as opposed to
      handling tensor/variable/model lifecycle internally.

    * **objects (list of tensors, variables, or models, optional)**
      When the worker is NOT a client worker, it stores all tensors
      it receives or creates in this dictionary.
      The key to each object is it's id.

    * **tmp_objects (list of tensors, variables, or models, optional)**
      When the worker IS a client worker, it stores some tensors temporarily
      in this _tmp_objects simply to ensure that they do not get deallocated by
      the Python garbage collector while in the process of being registered.
      This dictionary can be emptied using the clear_tmp_objects method.

    * **known_workers (list of **:class:`BaseWorker` ** objects, optional)** This dictionary
      can include all known workers.

    * **verbose (bool, optional)** A flag for whether or not to print events to stdout.

    :Example:

    >>> from syft.core.hooks import TorchHook
    >>> from syft.core.hooks import torch
    >>> from syft.core.workers import VirtualWorker
    >>> hook = TorchHook()
    Hooking into Torch...
    Overloading complete.
    >>> local = hook.local_worker
    >>> remote = VirtualWorker(id=1, hook=hook)
    >>> local.add_worker(remote)
    >>> x = torch.FloatTensor([1,2,3,4,5])
    >>> x
     1
     2
     3
     4
     5
    [torch.FloatTensor of size 5]
    >>> x.send(remote)
    >>> x
    [torch.FloatTensor - Locations:[<syft.core.workers.VirtualWorker object at 0x11848bda0>]]
    >>> x.get()
    >>> x
     1
     2
     3
     4
     5
    [torch.FloatTensor of size 5]
    """

    def __init__(
        self,  hook=None, id=0, is_client_worker=False, objects={},
        tmp_objects={}, known_workers={}, verbose=False, queue_size=0,
    ):

        super().__init__(
            hook=hook, id=id, is_client_worker=is_client_worker,
            objects=objects, tmp_objects=tmp_objects,
            known_workers=known_workers, verbose=verbose, queue_size=queue_size,
        )

    def _send_msg(self, message_wrapper_json_binary, recipient):
        """Sends a string message to another worker with message_type information
        indicating how the message should be processed.

        :Parameters:

        * **recipient (** :class:`VirtualWorker` **)** the worker being sent a message.

        * **message_wrapper_json_binary (binary)** the message being sent encoded in binary

        * **out (object)** the response from the message being sent. This can be a variety
          of object types. However, the object is typically only used during testing or
          local development with :class:`VirtualWorker` workers.
        """

        return recipient.receive_msg(message_wrapper_json_binary)
