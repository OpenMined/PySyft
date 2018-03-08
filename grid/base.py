from grid.lib import utils, keras_utils
from grid import channels
import base64
import json
import numpy as np
from threading import Thread
import torch
import time
import sys
from colorama import Fore, Back, Style
from bitcoin import base58


class PubSub(object):
    def __init__(self, mode, ipfs_addr='127.0.0.1', port=5001):
        self.api = utils.get_ipfs_api()
        peer_id = self.api.config_show()['Identity']['PeerID']
        self.id = f'{peer_id}'
        # switch to this to make local develop work
        # self.id = f'{mode}:{peer_id}'
        self.subscribed_list = []

    
    """
    Grid Tree Implementation

    Methods for Grid tree down here
    """

    def send_model(self, name, model_addr):
        task = utils.load_task(name)

        update = {
            'name': name,
            'model': model_addr,
            'task': task['address'],
            'creator': self.id,
            'parent': task['address']
        }

        update_addr = self.api.add_json(update)
        self.publish(channels.add_model(name), update_addr)

        print("SENDING MODEL!!!!")

    def add_model(self, name, model, parent=None):
        """
        Propose a model as a solution to a task.

        parent  - The name of the task.  e.g. MNIST
        model - A keras model. Down the road we should support more frameworks.
        """
        task = utils.load_task(name)
        p = None
        if parent is None:
            p = task['address']
        else:
            p = parent

        model_addr = keras_utils.keras2ipfs(model)

        update = {
            'name': name,
            'model': model_addr,
            'task': task['address'],
            'creator': self.id,
            'parent': p
        }

        update_addr = self.api.add_json(update)
        self.publish(channels.add_model(name), update_addr)
        print(f"ADDED NEW MODELS WEIGHT TO {update_addr}")
