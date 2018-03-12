from ..lib import utils, output_pipe, keras_utils
from .. import channels
import json
import threading
from .base import BaseService


class FitWorkerService(BaseService):

    # The purpose of this service is to train a model that was called using
    # the "fit" method - which is a more rigid training spec inspired by
    # sci-kit learn and keras.

    def __init__(self, worker):
        super().__init__(worker)

        self.worker.listen_to_channel(channels.openmined, self.fit_worker)

    def fit_worker(self, message):
        """
        When anyone broadcasts to the "openmined" channel, this method
        responds and routes the job to the appropriate framework
        """

        decoded = json.loads(message['data'])

        # if this node is the one that was requested for the job
        # (or the client simply doesn't care)
        if ((decoded['preferred_node'] == 'first_available')
                or (decoded['preferred_node'] == self.worker.id)):

            if (decoded['framework'] == 'keras'):
                return self.fit_keras(decoded)
            else:
                raise NotImplementedError(
                    "Only compatible with Keras at the moment")

    def fit_keras(self, decoded):
        """
        This method trains a Keras model using params that the client specifies.

        Clients can specify what data to use to train, validate, how many
        epochs to use, etc.

        Long term, the client should be able to specify anything that is used
        in keras model.fit
        """

        # loads keras model from ipfs
        model = keras_utils.ipfs2keras(self.api, decoded['model_addr'])

        # gets dataset from ipfs
        try:
            np_strings = json.loads(self.worker.api.cat(decoded['data_addr']))
        except NotImplementedError:
            raise NotImplementedError(
                "The IPFS API only supports Python 3.6. Please modify your environment."
            )

        # get input/validation data from ipfs
        input, target, valid_input, valid_target = list(
            map(lambda x: utils.deserialize_numpy(x), np_strings))

        # sets up channel for sending logging information back to the client
        # (so that they can see incremental progress)
        train_channel = decoded['train_channel']

        # Output pipe is a keras callback
        # https://keras.io/callbacks/
        # See `OutputPipe` for more info.
        self.worker.learner_callback = output_pipe.OutputPipe(
            api=self.worker.api,
            id=self.worker.id,
            publisher=self.worker.publish,
            channel=train_channel,
            epochs=decoded['epochs'],
            model_addr=decoded['model_addr'],
            model=model,
            email=self.worker.email)

        # When you train a model, you talk about it on a subchannel.
        # Start listening on this channel for updates about training.
        _args = (train_channel + ':' + self.worker.id, self.train_meta)
        monitor_thread = threading.Thread(
            target=self.worker.listen_to_channel, args=_args)
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
            callbacks=[self.worker.learner_callback])

        print('done')

    def train_meta(worker, message):
        """
        This method is used the handle meta data commands that can be passed
        between client and worker during training.

        The client can send the worker an `op_code` that will tell it to do something.
        """

        decoded = json.loads(message['data'])
        if 'op_code' not in decoded:
            return

        # The only currently supported op_code is to tell the worker to quit.
        # This is used when a node is too slow to train.  E.g. if two workers
        # pick up the same job, and worker A is much faster than worker B.
        # Then the client will tell worker B to quit.
        worker.learner_callback.stop_training = decoded['op_code'] == 'quit'
