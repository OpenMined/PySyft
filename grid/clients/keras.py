from . import base
from ..lib import utils
from ..lib import serde
from ..lib import coinbase_helper
import json
import random

class KerasClient(base.BaseClient):

    def __init__(self,min_om_nodes=1,known_workers=list(),include_github_known_workers=True,verbose=True):
        super().__init__(min_om_nodes=min_om_nodes,
                        known_workers=known_workers,
                        include_github_known_workers=include_github_known_workers,
                        verbose=verbose)

        self.email = None
        self.time_taken = None
        self.cb_helper = None

    def set_coinbase_api(self, api_key, api_secret):
        self.cb_helper = coinbase_helper.CoinbaseHelper(api_key, api_secret)

    def fit(self, model, input, target, valid_input=None, valid_target=None, batch_size=1, epochs=1, log_interval=1, message_handler=None, preferred_node='random'):

        if('p2p-circuit' in preferred_node or '/' in preferred_node):
            preferred_node = preferred_node.split("/")[-1]

        if(preferred_node == 'random'):
            nodes = self.get_openmined_nodes()
            preferred_node = nodes[random.randint(0,len(nodes)-1)]

        print("PREFERRED NODE:" + str(preferred_node))

        if(message_handler is None):
            message_handler = self.receive_model
        self.spec = self.generate_fit_spec(model=model,input=input,target=target,valid_input=valid_input,valid_target=valid_target,batch_size=batch_size,epochs=epochs,log_interval=log_interval,preferred_node=preferred_node)
        self.publish('openmined', self.spec)

        self.listen_to_channel_sync(self.spec['train_channel'], message_handler)

        if self.cb_helper and self.email and self.time_taken:
            self.cb_helper.send_ether(self.email, self.time_taken)

        return self.load_model(self.spec['model_addr']), self.spec

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

    def generate_fit_spec(self, model,input,target,valid_input=None,valid_target=None,batch_size=1,epochs=1,log_interval=1, framework = 'keras', model_class = None,preferred_node='first_available'):

        model_bin = utils.serialize_keras_model(model)
        model_addr = self.api.add_bytes(model_bin)

        if model_class is not None:
            self.api.add_bytes(model_class)

        train_input = utils.serialize_numpy(input)
        train_target = utils.serialize_numpy(target)

        if(valid_input is None):
            valid_input = utils.serialize_numpy(input)
        else:
            valid_input = utils.serialize_numpy(valid_input)

        if(valid_target is None):
            valid_target = utils.serialize_numpy(target)
        else:
            valid_target = utils.serialize_numpy(valid_target)

        datasets = [train_input, train_target, valid_input, valid_target]
        data_json = json.dumps(datasets)
        data_addr = self.api.add_str(data_json)

        spec = {}
        spec['type'] = "fit"
        spec['model_addr'] = model_addr
        spec['data_addr'] = data_addr
        spec['batch_size'] = batch_size
        spec['epochs'] = epochs
        spec['log_interval'] = log_interval
        spec['framework'] = framework
        spec['train_channel'] = 'openmined_train_' + str(model_addr)
        spec['preferred_node'] = preferred_node

        return spec

    def load_model(self, addr):
        return utils.ipfs2keras(addr)

    def receive_model(self, message, verbose=True):
        msg = json.loads(message['data'])

        if(msg is not None):
            if(msg['type'] == 'transact'):
                if('worker_email' in msg.keys()):
                    self.email = msg['worker_email']
                if('time_taken' in msg.keys()):
                    self.time_taken = msg['time_taken']

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
