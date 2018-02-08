from grid import ipfsapi
from grid.lib import OutputPipe, utils
from . import base

from torch.autograd import Variable
from colorama import Fore, Back, Style

import keras
import json
import numpy as np
import torch
import threading
import sys
import asyncio

"""
TODO: modify Client to store the source code for the model in IPFS.
            (think through logistics; introduces hurdles for packaging model source code)
TODO: figure out a convenient way to make robust training procedure for torch -- will probably want to use ignite for this
"""


class Worker(base.PubSub):

    def train_meta(self, message):
        decoded = json.loads(message['data'])
        if not 'op_code' in decoded:
            return

        self.learner_callback.stop_training = decoded['op_code'] == 'quit'

    # TODO: torch
    def fit_worker(self,message):

        decoded = json.loads(message['data'])

        if(decoded['framework'] == 'keras'):

            model = utils.ipfs2keras(decoded['model_addr'])

            try:
                np_strings = json.loads(self.api.cat(decoded['data_addr']))
            except:
                raise NotImplementedError("The IPFS API only supports Python 3.6. Please modify your environment.")

            input,target,valid_input,valid_target = list(map(lambda x:self.deserialize_numpy(x),np_strings))
            train_channel = decoded['train_channel']

            self.learner_callback = OutputPipe(
                id=self.id,
                publisher=self.publish,
                channel=train_channel,
                epochs=decoded['epochs'],
                model_addr=decoded['model_addr'],
                model=model
            )

            monitor_thread = threading.Thread(target = self.listen_to_channel, args = (self.train_meta, train_channel + ':' + self.id))
            monitor_thread.start()

            print('training model')

            model.fit(
                input,
                target,
                batch_size=decoded['batch_size'],
                validation_data=(valid_input, valid_target),
                verbose=False,
                epochs=decoded['epochs'],
                callbacks=[self.learner_callback]
            )

        else:
            raise NotImplementedError("Only compatible with Keras at the moment")


    def work(self):
        self.listen_to_channel(self.fit_worker, 'openmined')
