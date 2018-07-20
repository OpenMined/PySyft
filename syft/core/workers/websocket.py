import websockets
import asyncio
import json
from .. import utils

from .base import BaseWorker


class WebSocketWorker(BaseWorker):
    '''
    A socket for the worker but using websocket library
    instead of socket library.
    '''

    def __init__(self,  hook=None, hostname='localhost', port=8110, max_connections=5,
                 id=0, is_client_worker=True, objects={}, tmp_objects={},
                 known_workers={}, verbose=True, is_pointer=False, queue_size=0):

        #Initializing the parent class
        super().__init__(hook=hook, id=id, is_client_worker=is_client_worker,
                         objects=objects, tmp_objects=tmp_objects,
                         known_workers=known_workers, verbose=verbose, queue_size=queue_size)

        self.hostname = hostname
        self.port = port
        self.max_connections = max_connections
        self.is_pointer = is_pointer
        self.uri = 'ws://' + str(self.hostname) + ':' + str(self.port)


        if(self.is_pointer):
            if(self.verbose):
                print("Attaching Pointer to Socket Worker...")

            clientsocket = websockets.connect(self.uri)
            self.clientsocket = clientsocket

        else:

            if(self.verbose):
                print("Starting Socket Worker...")

            self._run_server_socket()

            # if it's a client worker, then we don't want it waiting for commands
            # because it's going to be issuing commands.
            if(not is_client_worker or self.is_pointer):
                print("Ready to receive commands...")
                # self._listen()
            else:
                print("Ready!")

    async def _run_server_socket(self):
        self.serversocket = websockets.serve(self._server_socket_run, self.hostname, self.port)
        asyncio.get_event_loop().run_until_complete(self.serversocket)
        asyncio.get_event_loop().run_forever()

    async def _client_socket_connect(self, json_request):
        async with websockets.connect(self.uri) as client_socket:
            await client_socket.send(json_request)
            return await client_socket.recv()


    async def _server_socket_run(self, websocket, path):
        message_wrapper_json = await websocket.recv()
        decoder = utils.PythonJSONDecoder(self)
        message_wrapper = decoder.decode(message_wrapper_json)
        await websocket.send(self.process_message_type(message_wrapper))



    async def _client_socket_run(self, json_request):
        return asyncio.get_event_loop().run_until_complete(self._client_socket_connect(json_request))

    def whoami(self):
        return json.dumps({"hostname": self.hostname, "port": self.port, "id": self.id})


    async def _send_msg(self, message_wrapper_json_binary, recipient):
        """Sends a string message to another worker with message_type information
        indicating how the message should be processed.

        :Parameters:

        * **recipient (** :class:`VirtualWorker` **)** the worker being sent a message.

        * **message_wrapper_json_binary (binary)** the message being sent encoded in binary

        * **out (object)** the response from the message being sent. This can be a variety
          of object types. However, the object is typically only used during testing or
          local development with :class:`VirtualWorker` workers.
        """


        await self._client_socket_run(message_wrapper_json_binary)
        response = self._process_buffer(recipient.clientsocket)

        return response

    @classmethod
    async def _process_buffer(cls, socket, buffer_size=1024, delimiter="\n"):
        # WARNING: will hang if buffer doesn't finish with newline
        buffer = await socket.recv().decode('utf-8')
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


    async def listen(self, num_messages=-1):
        """
        Starts SocketWorker server on the correct port and handles message as they
        are received.
        """
        while num_messages != 0:
            connection = self.serversocket()
            address, port = connection.remote_address()
            try:
                while num_messages != 0:
                    # collapse buffer of messages into a string
                    message = self._process_buffer(self.serversocket)

                    # process message and generate response
                    response = self._client_socket_run(message)

                    if(response[-1] != "\n"):
                        response += "\n"
                    # send response back

                    await connection.send(response.encode())

                    if(self.verbose):
                        print("Received Command From:", self.uri)

                    num_messages -= 1
            finally:
                connection.close()
