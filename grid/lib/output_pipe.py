import keras
from . import utils
from time import sleep


class OutputPipe(keras.callbacks.Callback):

    def __init__(self, id, publisher, channel, epochs, model_addr, model):
        self.id = id
        self.publisher = publisher
        self.channel = channel
        self.epochs = epochs
        self.model_addr = model_addr
        self.model = model

        self.stop_training = False

    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, epoch, logs={}):
        acc = logs.get('acc')
        loss = logs.get('loss')

        spec = {}
        spec['type'] = 'log'
        spec['eval_loss'] = loss
        spec['epoch_id'] = epoch
        spec['num_epochs'] = self.epochs
        spec['parent_model'] = self.model_addr
        spec['worker_id'] = self.id

        self.model.stop_training = self.stop_training
        self.publisher(channel=self.channel, dict_message=spec)

    def on_train_end(self, logs={}):
        spec = {}
        spec['type'] = 'transact'
        spec['model_addr'] = utils.keras2ipfs(self.model)
        spec['eval_loss'] = logs.get('loss')
        spec['parent_model'] = self.model_addr
        spec['worker_id'] = self.id

        self.publisher(channel=self.channel, dict_message=spec)
