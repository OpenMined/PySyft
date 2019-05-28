import tf_encrypted as tfe



class TensorflowClientWorker:
    def connect_to_model(self, config_filename, input_shape, output_shape):

        config = tfe.RemoteConfig.load(config_filename)
        tfe.set_config(config)

        # We set the protocol to Pond. The protocol
        # selected shouldn't matter on the client side
        tfe.set_protocol(tfe.protocol.Pond())

        self._tf_client = tfe.serving.QueueClient(
            input_shape=input_shape,
            output_shape=output_shape)

        sess = tfe.Session(config=config)
        self._tf_session = sess


    def query_model(self, data):

        return self._tf_client.run(self._tf_session, data)
