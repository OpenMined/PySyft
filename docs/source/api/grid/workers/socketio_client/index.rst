:mod:`grid.workers.socketio_client`
===================================

.. py:module:: grid.workers.socketio_client


Module Contents
---------------

.. py:class:: WebsocketIOClientWorker(hook, host: str, port: int, id: Union[int, str] = 0, is_client_worker: bool = False, log_msgs: bool = False, verbose: bool = False, data: List[Union[torch.Tensor, AbstractTensor]] = None)

   Bases: :class:`syft.workers.virtual.VirtualWorker`

   A worker that forwards a message to a SocketIO server and wait for its response.

   This client then waits until the server returns with a result or an ACK at which point it finishes the
   _recv_msg operation.

   .. method:: _send_msg(self, message: bin)



   .. method:: _recv_msg(self, message: bin)



   .. method:: connect(self)



   .. method:: disconnect(self)




