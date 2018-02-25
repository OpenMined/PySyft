from . import base_worker
from ...lib import strings
from .. import channels
import json

class GridCompute(base_worker.GridWorker):

    """
    This class runs a worker whose purpose is to do the following:
       - PRIMARY: use local compute resources to train models at the request of clients on the network
       - SECONDARY: learn about the existence of other nodes on the network - and help others to do so when asked
    """


    def __init__(self):
        super().__init__()  

        # prints a pretty picture of a Computer
        print(strings.compute)

        # This process listens for models that it can train.
        self.listen_to_channel(channels.openmined, self.fit_worker)

    def train_meta(self, message):
        """
        TODO: describe the purpose of this method.
        """

        decoded = json.loads(message['data'])
        if 'op_code' not in decoded:
            return

        self.learner_callback.stop_training = decoded['op_code'] == 'quit'


    # TODO: torch
    def fit_worker(self, message):

        """
        When anyone broadcasts to the "openmined" channel, this method responds and routes the job to the appropriate framework
        """

        decoded = json.loads(message['data'])

        # if this node is the one that was requested for the job (or the client simply doesn't care)
        if((decoded['preferred_node'] == 'first_available') or (decoded['preferred_node'] == self.id)):

            if(decoded['framework'] == 'keras'):
                return self.fit_keras(decoded)
            else:
                raise NotImplementedError("Only compatible with Keras at the moment")


    def fit_keras(self,decoded):

        """
        This method trains a Keras model according to the insttuc
        """

        # loads keras model from ipfs
        model = utils.ipfs2keras(decoded['model_addr'])

        # gets dataset from ipfs
        try:
            np_strings = json.loads(self.api.cat(decoded['data_addr']))
        except NotImplementedError:
            raise NotImplementedError("The IPFS API only supports Python 3.6. Please modify your environment.")

        input, target, valid_input, valid_target = list(map(lambda x: self.deserialize_numpy(x),np_strings))

        # sets up channel for sending logging information back to the client (so that they can see incremental progress)
        train_channel = decoded['train_channel']

        self.learner_callback = OutputPipe(
            id=self.id,
            publisher=self.publish,
            channel=train_channel,
            epochs=decoded['epochs'],
            model_addr=decoded['model_addr'],
            model=model
        )

        _args = (self.train_meta, train_channel + ':' + self.id)
        monitor_thread = threading.Thread(target=self.listen_to_channel,
                                          args=_args)
        monitor_thread.start()

        print('training model')

        # trains model
        model.fit(
            input,
            target,
            batch_size=decoded['batch_size'],
            validation_data=(valid_input, valid_target),
            verbose=False,
            epochs=decoded['epochs'],
            callbacks=[self.learner_callback]
        )

        print('done')