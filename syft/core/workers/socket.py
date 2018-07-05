import socket

from .base import BaseWorker


class SocketWorker(BaseWorker):
    """
    A worker capable of performing the functions of a BaseWorker across
    a socket connection. Note that the worker is NOT responsible for
    creating the socket connection.

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

    :Example Server:

    >>> from syft.core.hooks import TorchHook
    >>> from syft.core.workers import VirtualWorker, SocketWorker
    >>> hook = TorchHook()
    Hooking into Torch...
    Overloading complete.
    >>> local_worker = SocketWorker(hook=hook,
                            id=2,
                            port=8181,
                            is_pointer=False,
                            is_client_worker=False)
    Starting Socket Worker...
    Ready to receive commands...

    :Example Client:

    >>> import torch
    >>> from syft.core.hooks import TorchHook
    >>> from syft.core.workers import SocketWorker
    >>> hook = TorchHook(local_worker=SocketWorker(id=0, port=8182))
    Starting Socket Worker...
    Ready!
    Hooking into Torch...
    Overloading complete.
    >>> remote_client = SocketWorker(hook=hook,id=2, port=8181, is_pointer=True)
    >>> hook.local_worker.add_worker(remote_client)
    Attaching Pointer to Socket Worker...
    >>> x = torch.FloatTensor([1,2,3,4,5]).send(remote_client)
    >>> x2 = torch.FloatTensor([1,2,3,4,4]).send(remote_client)
    >>> y = x + x2 + x
    >>> y
    [torch.FloatTensor - Locations:[<syft.core.workers.SocketWorker object at 0x7f94eaaa6630>]]
    >>> y.get()
      3
      6
      9
     12
     14
    [torch.FloatTensor of size 5]
    """

    def __init__(self,  hook=None, hostname='localhost', port=8110, max_connections=5,
                 id=0, is_client_worker=True, objects={}, tmp_objects={},
                 known_workers={}, verbose=True, is_pointer=False, queue_size=0):

        super().__init__(hook=hook, id=id, is_client_worker=is_client_worker,
                         objects=objects, tmp_objects=tmp_objects,
                         known_workers=known_workers, verbose=verbose, queue_size=queue_size)

        self.hostname = hostname
        self.port = port

        self.max_connections = max_connections
        self.is_pointer = is_pointer

        if(self.is_pointer):
            if(self.verbose):
                print("Attaching Pointer to Socket Worker...")
            self.serversocket = None

            clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            clientsocket.connect((self.hostname, self.port))
            self.clientsocket = clientsocket

        else:

            if(self.verbose):
                print("Starting Socket Worker...")
            self.serversocket = socket.socket(
                socket.AF_INET, socket.SOCK_STREAM)
            self.serversocket.bind((self.hostname, self.port))

            # become a server socket, maximum 5 connections
            self.serversocket.listen(self.max_connections)

            # if it's a client worker, then we don't want it waiting for commands
            # because it's going to be issuing commands.
            if(not is_client_worker or self.is_pointer):
                print("Ready to receive commands...")
                self._listen()
            else:
                print("Ready!")

    def _listen(self):
        """
        Starts SocketWorker server on the correct port and handles message as they
        are received.
        """
        while True:

            # blocking until a message is received
            connection, address = self.serversocket.accept()
            try:
                while True:
                    # collapse buffer of messages into a string
                    message = self._process_buffer(connection)

                    # process message and generate response
                    response = self.receive_msg(message, False)

                    # send response back
                    connection.send(response.encode())

                    if(self.verbose):
                        print("Received Command From:", address)
            finally:
                connection.close()

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

        recipient.clientsocket.send(message_wrapper_json_binary)

        response = self._process_buffer(recipient.clientsocket)

        return response

    @classmethod
    def _process_buffer(cls, socket, buffer_size=1024, delimiter="\n"):
        # WARNING: will hang if buffer doesn't finish with newline

        buffer = socket.recv(buffer_size).decode('utf-8')
        buffering = True
        while buffering:
            if delimiter in buffer:
                (line, buffer) = buffer.split(delimiter, 1)
                return line + delimiter
            else:
                more = socket.recv(buffer_size).decode('utf-8')
                if not more:
                    buffering = False
                else:
                    buffer += more
        if buffer:
            return buffer
