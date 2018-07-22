import websockets
import asyncio
import json

from .. import utils
from .base import BaseWorker

class WebSocketWorker(BaseWorker):
    """
    A worker capable of performing the functions of a BaseWorker across
    a websocket connection. This worker essentially can replace Socket
    Worker.

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


    """

    def __init__(self,  hook=None, hostname='localhost', port=8110, max_connections=5,
                 id=0, is_client_worker=True, objects={}, tmp_objects={},
                 known_workers={}, verbose=True, is_pointer=False, queue_size=0):

        super().__init__(hook=hook, id=id, is_client_worker=is_client_worker,
                         objects=objects, tmp_objects=tmp_objects,
                         known_workers=known_workers, verbose=verbose, queue_size=queue_size)

        self.is_asyncronous = True
        self.hook = hook
        self.hostname = hostname
        self.port = port
        self.uri = "ws://" + self.hostname + ":" + str(self.port)

        self.max_connections = max_connections
        self.is_pointer = is_pointer

        if (self.is_pointer):
            if (self.verbose):
                print("Attaching Pointer to WebSocket Worker....")
            self.serversocket = None
            clientsocket = websockets.client.connect(self.uri)
            self.clientsocket = clientsocket

        else:
            if (self.verbose):
                print("Starting a Websocket Worker....")
                if (not is_client_worker or self.is_pointer):
                    print("Ready to recieve commands....")
                    self.serversocket = websockets.serve(self._server_socket_listener,
                                                            self.hostname, self.port)
                    print('Server Socket has been initialized')
                    asyncio.get_event_loop().run_until_complete(self.serversocket)
                    asyncio.get_event_loop().run_forever()

                else:
                    print("Ready...")


    async def _client_socket_connect(self, json_request):
        """
        Establishes a connection to the server socket and waits for a response.
        Then the response is returned.

        :Parameters:

        * **json_request** JSON request that is needed to be sent to the server Socket

        * ** out (json)** The response from the server is returned as JSON.
        """


        async with websockets.connect(self.uri) as client_socket:
            await client_socket.send(json_request)
            recieved_msg = await client_socket.recv()
            return recieved_msg

    async def _server_socket_listener(self, websocket, path):
        msg_wrapper_byte = await websocket.recv()
        msg_wrapper_str = msg_wrapper_byte.decode('utf-8')
        if (self.verbose):
            print("Recieved Command From:", self.uri)
        decoder = utils.PythonJSONDecoder(self)
        msg_wrapper = decoder.decode(msg_wrapper_str)
        await websocket.send(self.process_message_type(msg_wrapper))

    def whoami(self):
        return json.dumps({"uri": self.uri, "id": self.id})

    async def _send_msg(self, message_wrapper_json_binary, recipient):
        response = await recipient._client_socket_listener(message_wrapper_json_binary)
        response = self._process_buffer(response=response)
        return response

    def send_msg(self, message, message_type, recipient):
        """Sends a string message to another worker with message_type information
        indicating how the message should be processed.

        :Parameters:

        * **recipient (** :class:`VirtualWorker` **)** the worker being sent a message.

        * **message (string)** the message being sent

        * **message_type (string)** the type of message being sent. This affects how
          the message is processed by the recipient. The types of message are described
          in :func:`receive_msg`.

        * **out (object)** the response from the message being sent. This can be a variety
          of object types. However, the object is typically only used during testing or
          local development with :class:`VirtualWorker` workers.
        """
        message_wrapper = {}
        message_wrapper['message'] = message
        message_wrapper['type'] = message_type
        self.message_queue.append(message_wrapper)
        if self.queue_size:
            if len(self.message_queue) > self.queue_size:
                message_wrapper = self.compile_composite_message()
            else:
                return None

        message_wrapper_json = json.dumps(message_wrapper) + "\n"

        message_wrapper_json_binary = message_wrapper_json.encode()

        self.message_queue = []
        response = recipient._client_socket_listener(message_wrapper_json_binary)
        response = self._process_buffer(response=response)
        return response

    def _process_buffer(cls, response, delimiter="\n"):
        buffer = response
        buffering = True
        if delimiter in buffer:
            (line, buffer) = buffer.split(delimiter, 1)
            return line + delimiter
        else:
            return buffer

    def _client_socket_listener(cls, message_wrapper_json_binary):
        response = asyncio.get_event_loop().run_until_complete(
            cls._client_socket_connect(message_wrapper_json_binary))
        return response


