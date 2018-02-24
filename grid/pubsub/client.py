from grid.lib import utils
from grid.pubsub.base import PubSub
from grid.pubsub import channels, commands
from bitcoin import base58
from colorama import Fore, Back, Style
import ipywidgets as widgets
import json
import sys
import os


class Client(PubSub):
    def __init__(self,min_om_nodes=1,known_workers=list(),include_github_known_workers=True):
        super().__init__('client')
        self.progress = {}

        if(include_github_known_workers):
            import requests

            workers = requests.get('https://github.com/OpenMined/BootstrapNodes/raw/master/known_workers').text.split("\n")
            for w in workers:
                if('p2p-circuit' in w):
                    known_workers.append(w)

        known_workers = list(set(known_workers))

        if(len(known_workers) > 0):
            print(f'\n{Fore.BLUE}UPDATE: {Style.RESET_ALL}Querying known workers...')    
            for worker in known_workers:
                try:
                    sys.stdout.write('\tWORKER: ' + str(worker) + '...')    
                    self.api.swarm_connect(worker)
                    sys.stdout.write(f'{Fore.GREEN}SUCCESS!!!{Style.RESET_ALL}\n')    
                except:
                    sys.stdout.write(f'{Fore.RED}FAIL!!!{Style.RESET_ALL}\n')
                    ""

        self.listen_for_openmined_nodes(min_om_nodes)

    def fit(self, model, input, target, valid_input=None, valid_target=None, batch_size=1, epochs=1, log_interval=1, message_handler=None):
        if(message_handler is None):
            message_handler = self.receive_model
        self.spec = self.generate_fit_spec(model,input,target,valid_input,valid_target,batch_size,epochs,log_interval)
        self.publish('openmined', self.spec)

        self.listen_to_channel_sync(self.spec['train_channel'], message_handler)
        return self.spec

    def update_progress(self, parent_model, worker_id, num_epochs, epoch_id):
        if parent_model not in self.progress:
            self.progress[parent_model] = {}

        if worker_id not in self.progress[parent_model]:
            self.progress[parent_model][worker_id] = 0

        p = epoch_id / num_epochs
        self.progress[parent_model][worker_id] = p

        return p

    def max_progress(self, parent_model):
        if parent_model not in self.progress:
            return 0

        max_progress = 0
        for worker_id, progress in self.progress[parent_model].items():
            if progress > max_progress:
                max_progress = progress

        return max_progress

    # TODO: torch
    def receive_model(self, message, verbose=True):
        msg = json.loads(message['data'])

        if(msg is not None):
            if(msg['type'] == 'transact'):
                return utils.ipfs2keras(msg['model_addr']), msg
            elif(msg['type'] == 'log'):
                if(verbose):
                    output = "Worker:" + msg['worker_id'][-5:]
                    output += " - Epoch " + str(msg['epoch_id']) + " of " + str(msg['num_epochs'])
                    output += " - Valid Loss: " + str(msg['eval_loss'])[0:8]
                    print(output)

                # Figure out of we should tell this worker to quit.
                parent_model = msg['parent_model']
                worker_id = msg['worker_id']
                num_epochs = msg['num_epochs']
                epoch_id = msg['epoch_id']

                progress = self.update_progress(parent_model, worker_id,
                                                num_epochs, epoch_id)
                max_progress = self.max_progress(parent_model)

                if progress < max_progress * 0.75:
                    quit = {}
                    quit['op_code'] = 'quit'
                    self.publish(self.spec['train_channel'] + ':' + worker_id,
                                 quit)

    # TODO: framework = 'torch'
    def generate_fit_spec(self, model,input,target,valid_input=None,valid_target=None,batch_size=1,epochs=1,log_interval=1, framework = 'keras', model_class = None):

        model_bin = utils.serialize_keras_model(model)
        model_addr = self.api.add_bytes(model_bin)

        if model_class is not None:
            self.api.add_bytes(model_class)

        train_input = self.serialize_numpy(input)
        train_target = self.serialize_numpy(target)

        if(valid_input is None):
            valid_input = self.serialize_numpy(input)
        else:
            valid_input = self.serialize_numpy(valid_input)

        if(valid_target is None):
            valid_target = self.serialize_numpy(target)
        else:
            valid_target = self.serialize_numpy(valid_target)

        datasets = [train_input, train_target, valid_input, valid_target]
        data_json = json.dumps(datasets)
        data_addr = self.api.add_str(data_json)

        spec = {}
        spec['model_addr'] = model_addr
        spec['data_addr'] = data_addr
        spec['batch_size'] = batch_size
        spec['epochs'] = epochs
        spec['log_interval'] = log_interval
        spec['framework'] = framework
        spec['train_channel'] = 'openmined_train_' + str(model_addr)
        return spec

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
