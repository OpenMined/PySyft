:mod:`grid.websocket_client`
============================

.. py:module:: grid.websocket_client


Module Contents
---------------

.. data:: MODEL_LIMIT_SIZE
   

   

.. py:class:: WebsocketGridClient(hook, address, id: Union[int, str] = 0, auth: dict = None, is_client_worker: bool = False, log_msgs: bool = False, verbose: bool = False, chunk_size: int = MODEL_LIMIT_SIZE)

   Bases: :class:`syft.WebsocketClientWorker`, :class:`syft.federated.federated_client.FederatedClient`

   Websocket Grid Client.

   .. method:: url(self)
      :property:


      Get Node URL Address.

      :returns: Node's address.
      :rtype: address (str)


   .. method:: _update_node_reference(self, new_id: str)


      Update worker references changing node id references at hook structure.

      :param new_id: New worker ID.
      :type new_id: str


   .. method:: parse_address(self, address: str)


      Parse Address string to define secure flag and split into host and port.

      :param address: Adress of remote worker.
      :type address: str


   .. method:: get_node_id(self)


      Get Node ID from remote node worker

      :returns: node id used by remote worker.
      :rtype: node_id (str)


   .. method:: connect_nodes(self, node)


      Connect two remote workers between each other.
      If this node is authenticated, use the same credentials to authenticate the candidate node.

      :param node: Node that will be connected with this remote worker.
      :type node: WebsocketGridClient

      :returns: node response.
      :rtype: node_response (dict)


   .. method:: authenticate(self, user: Union[str, dict])


      Perform Authentication Process using credentials grid credentials.
      Grid credentials can be loaded calling the function gr.load_credentials().

      :param user: String containing the username of a loaded credential or a credential's dict.

      :raises RuntimeError: If authentication process fail.


   .. method:: _forward_json_to_websocket_server_worker(self, message: dict)


      Prepare/send a JSON message to a remote grid node and receive the response.

      :param message: message payload.
      :type message: dict

      :returns: response payload.
      :rtype: node_response (dict)


   .. method:: _forward_to_websocket_server_worker(self, message: bin)


      Prepare/send a binary message to a remote grid node and receive the response.
      :param message: message payload.
      :type message: bytes

      :returns: response payload.
      :rtype: node_response (bytes)


   .. method:: serve_model(self, model, model_id: str = None, allow_download: bool = False, allow_remote_inference: bool = False)


      Hosts the model and optionally serve it using a Socket / Rest API.

      :param model: A jit model or Syft Plan.
      :param model_id: An integer or string representing the model id used to retrieve the model
                       later on using the Rest API. If this is not provided and the model is a Plan
                       we use model.id, if the model is a jit model we raise an exception.
      :type model_id: str
      :param allow_download: If other workers should be able to fetch a copy of this model to run it locally set this to True.
      :type allow_download: bool
      :param allow_remote_inference: If other workers should be able to run inference using this model through a Rest API interface set this True.
      :type allow_remote_inference: bool

      :returns: True if model was served sucessfully, raises a RunTimeError otherwise.
      :rtype: result (bool)

      :raises ValueError: if model_id is not provided and model is a jit model (aka does not have an id attribute).
      :raises RunTimeError: if there was a problem during model serving.


   .. method:: run_remote_inference(self, model_id, data)


      Run a dataset inference using a remote model.

      :param model_id: Model ID.
      :type model_id: str
      :param data: dataset to be inferred.
      :type data: Tensor

      :returns: Inference result
      :rtype: inference (Tensor)

      :raises RuntimeError: If an unexpected behavior happen, It will forward the error message.


   .. method:: _return_bool_result(self, result, return_key=None)



   .. method:: _send_http_request(self, route, data, request, N: int = 10, unhexlify: bool = True, return_response_text: bool = True)


      Helper function for sending http request to talk to app.

      :param route: App route.
      :type route: str
      :param data: Data to be sent in the request.
      :type data: str
      :param request: Request type (GET, POST, PUT, ...).
      :type request: str
      :param N: Number of tries in case of fail. Default is 10.
      :type N: int
      :param unhexlify: A boolean indicating if we should try to run unhexlify on the response or not.
      :type unhexlify: bool
      :param return_response_text: If True return response.text, return raw response otherwise.
      :type return_response_text: bool

      :returns: If return_response_text is True return response.text, return raw response otherwise.
      :rtype: response (bool)


   .. method:: _send_streaming_post(self, route: str, data: dict = None)


      Used to send large models / datasets using stream channel.

      :param route: Service endpoint
      :type route: str
      :param data: dict with tensors / models to be uploaded.
      :type data: dict

      :returns: response from server
      :rtype: response (str)


   .. method:: _send_get(self, route, data=None, **kwargs)



   .. method:: models(self)
      :property:


      Get models stored at remote grid node.

      :returns: List of models stored in this grid node.
      :rtype: models (List)


   .. method:: delete_model(self, model_id: str)


      Delete a model previously registered.

      :param model_id: ID of the model that will be deleted.
      :type model_id: String

      :returns: If succeeded, return True.
      :rtype: result (bool)


   .. method:: download_model(self, model_id: str)


      Download a model to run it locally.

      :param model_id: ID of the model that will be downloaded.
      :type model_id: str

      :returns: Model to be downloaded.
      :rtype: model

      :raises RuntimeError: If an unexpected behavior happen, It will forward the error message.


   .. method:: serve_encrypted_model(self, encrypted_model: sy.messaging.plan.Plan)


      Serve a model in a encrypted fashion using SMPC.

      A wrapper for sending the model. The worker is responsible for sharing the model using SMPC.

      :param encrypted_model: A pĺan already shared with workers using SMPC.
      :type encrypted_model: syft.Plan

      :returns: True if model was served successfully, raises a RunTimeError otherwise.
      :rtype: result (bool)


   .. method:: __str__(self)




