from grid import ipfsapi
from grid.lib import OutputPipe, utils
from . import base

import keras
import json
import numpy as np
import torch
from torch.autograd import Variable
import sys
from colorama import Fore, Back, Style

"""
TODO: modify Client to store the source code for the model in IPFS.
            (think through logistics; introduces hurdles for packaging model source code)
TODO: figure out a convenient way to make robust training procedure for torch -- will probably want to use ignite for this
"""


class Worker(base.PubSub):

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

            pipe = OutputPipe(
                id=self.id,
                publisher=self.publish,
                channel=decoded['train_channel'],
                epochs=decoded['epochs'],
                model_addr=decoded['model_addr'],
                model=model
            )

            model.fit(
                input,
                target,
                batch_size=decoded['batch_size'],
                validation_data=(valid_input, valid_target),
                verbose=False,
                epochs=decoded['epochs'],
                callbacks=[pipe]
            )

        else:
            raise NotImplementedError("Only compatible with Keras at the moment")


    def work(self):
        self.listen_to_channel(self.fit_worker,'openmined')
