:mod:`grid.grid_network`
========================

.. py:module:: grid.grid_network


Module Contents
---------------

.. data:: SMPC_HOST_CHUNK
   :annotation: = 4

   

.. py:class:: GridNetwork(gateway_url)

   Bases: :class:`object`

   The purpose of the Grid Network class is to control the entire communication flow by abstracting operational steps.

   .. attribute:: - gateway_url

      network address to which you want to connect.

   .. attribute:: - connected_grid_nodes

      Grid nodes that are connected to the application.

   .. method:: search(self, *query)


      Search a set of tags across the grid network.

      :param query: A set of dataset tags.

      :returns: matrix of tensor pointers.
      :rtype: tensor_matrix


   .. method:: serve_encrypted_model(self, model)


      This method wiil choose some grid nodes at grid network to host an encrypted model.

      :param model: Model to be hosted.

      Raise:
          RuntimeError : If grid network doesn't have enough workers to host an encrypted model.


   .. method:: serve_model(self, model, model_id, allow_remote_inference: bool = False, allow_download: bool = False)


      This method will choose one of grid nodes registered in the grid network to host a plain text model.
      :param model: Model to be hosted.
      :param model_id: Model's ID.
      :param allow_remote_inference: Allow workers to run inference in this model.
      :param allow_download: Allow workers to copy the model and run it locally.


   .. method:: run_encrypted_inference(self, model_id, data, copy=True)


      Search for an encrypted model and perform inference.

      :param model_id: Model's ID.
      :param data: Dataset to be shared/inferred.
      :param copy: Boolean flag to perform encrypted inference without lose plan.

      :returns: Inference's result.
      :rtype: Tensor


   .. method:: run_remote_inference(self, model_id, data)


      This method will search for a specific model registered on grid network, if found,
      It will run inference.
      :param model_id: Model's ID.
      :param dataset: Data used to run inference.

      :returns: Inference's result.
      :rtype: Tensor


   .. method:: query_model(self, model_id)


      This method will search for a specific model registered on grid network, if found,
      It will return all grid nodes that contains the desired model.
      :param model_id: Model's ID.
      :param data: Data used to run inference.

      :returns: List of workers that contains the desired model.
      :rtype: workers


   .. method:: __connect_with_node(self, node_id, node_url)



   .. method:: disconnect_nodes(self)




