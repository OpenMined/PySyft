from grid.lib import utils
from grid.base import PubSub
from grid import channels, commands
from bitcoin import base58
from colorama import Fore, Back, Style
import ipywidgets as widgets
import json
import sys
import os
import random
from .services.listen_for_openmined_nodes import ListenForOpenMinedNodesService

class Client(PubSub):

    def __init__(self,min_om_nodes=1,known_workers=list(),include_github_known_workers=True):
        super().__init__('client')
        self.progress = {}

        self.processes = {}

        self.processes['listen_for_openmined_nodes'] = ListenForOpenMinedNodesService(self,min_om_nodes,include_github_known_workers)
        # self.listen_for_openmined_nodes(min_om_nodes,include_github_known_workers)

    

    

    # TODO: torch
    

    # TODO: framework = 'torch'
    

    """
    Grid Tree Implementation

    Methods for Grid tree down here
    """

    def found_task(self, message):
        fr = base58.encode(message['from'])

        tasks = json.loads(message['data'])
        for task in tasks:
            # utils.store_task(task['name'], task['address'])
            name = task['name']
            addr = task['address']

            hbox = widgets.HBox([widgets.Label(name), widgets.Label(addr)])
            self.all_tasks.children += (hbox, )


    def find_tasks(self):
        self.publish(channels.list_tasks, "None")
        self.all_tasks = widgets.VBox([widgets.HBox([widgets.Label('TASK NAME'), widgets.Label('ADDRESS')])])
        self.listen_to_channel(channels.list_tasks_callback(self.id), self.found_task)

        return self.all_tasks

    def add_task(self, name, data_dir=None, adapter=None):
        if data_dir == None and adapter == None:
            print(f'{Fore.RED}data_dir and adapter can not both be None{Style.RESET_ALL}')
            return

        task_data = {
            'name': name,
            'creator': self.id
        }

        if data_dir != None:
            task_data['data_dir'] = data_dir
        if adapter != None:
            with open(adapter, 'rb') as f:
                adapter_bin = f.read()
                f.close()
            adapter_addr = self.api.add_bytes(adapter_bin)
            task_data['adapter'] = adapter_addr

        addr = self.api.add_json(task_data)
        utils.store_task(name, addr)

        data = json.dumps([{'name': name, 'address': addr}])
        self.publish('openmined:add_task', data)

    def best_models(self, task):
        self.show_models = widgets.VBox([widgets.HBox([widgets.Label('Model Address')])])
        self.listen_to_channel(channels.add_model(task), self.__added_model)
        self.publish(channels.list_models, task)

        return self.show_models

    def __added_model(self, message):
        info = self.api.get_json(message['data'])
        model_addr = info['model']

        hbox = widgets.HBox([widgets.Label(model_addr)])
        self.show_models.children += (hbox,)

    def load_model(self, addr):
        return utils.ipfs2keras(addr)
