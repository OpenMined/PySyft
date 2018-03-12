import keras
from . import keras_utils
from time import time


class OutputPipe(keras.callbacks.Callback):
    """
    Output pipe is a keras callback (https://keras.io/callbacks/)

    This class hooks into training and is used to update clients on the progress
    of training (trying to mimick keras output going back to a client).

    Or, this class can be used to quit training if a client tells you to.
    """

    def __init__(self, api, id, publisher, channel, epochs, model_addr, model,
                 email):
        self.api = api
        self.id = id
        self.publisher = publisher
        self.channel = channel
        self.epochs = epochs
        self.model_addr = model_addr
        self.model = model

        self.stop_training = False

    def on_train_begin(self, logs={}):
        self.losses = []
        self.startTime = time()

    def on_epoch_end(self, epoch, logs={}):
        loss = logs.get('loss')

        spec = {}
        spec['type'] = 'log'
        spec['eval_loss'] = loss
        spec['epoch_id'] = epoch
        spec['num_epochs'] = self.epochs
        spec['parent_model'] = self.model_addr
        spec['worker_id'] = self.id

        # Quit training if the client tells you to
        self.model.stop_training = self.stop_training
        # Update client on the latest epoch
        self.publisher(channel=self.channel, message=spec)

    def on_train_end(self, logs={}):
        spec = {}
        spec['type'] = 'transact'
        spec['model_addr'] = keras_utils.keras2ipfs(self.api, self.model)
        spec['eval_loss'] = logs.get('loss')
        spec['parent_model'] = self.model_addr
        spec['worker_id'] = self.id

        # Tell the client you are finished training this model
        self.publisher(channel=self.channel, message=spec)
