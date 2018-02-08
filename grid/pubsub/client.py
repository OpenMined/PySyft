from grid import ipfsapi
from grid.lib import utils
from grid.pubsub.base import PubSub

import json

class Client(PubSub):

    def __init__(self):
        super().__init__()
        self.progress = {}

    def fit(self, model,input,target,valid_input=None,valid_target=None,batch_size=1,epochs=1,log_interval=1,message_handler=None):
        if(message_handler is None):
            message_handler = self.receive_model
        self.spec = self.generate_fit_spec(model,input,target,valid_input,valid_target,batch_size,epochs,log_interval)
        self.publish('openmined', self.spec)

        trained = self.listen_to_channel(message_handler, self.spec['train_channel'])
        return trained

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
    def receive_model(self,message, verbose=True):
        msg = json.loads(message['data'])

        if(msg is not None):
            if(msg['type'] == 'transact'):
                return utils.ipfs2keras(msg['model_addr']),msg
            elif(msg['type'] == 'log'):
                if(verbose):
                    output =  "Worker:" + msg['worker_id'][-5:]
                    output += " - Epoch " + str(msg['epoch_id']) + " of " + str(msg['num_epochs'])
                    output += " - Valid Loss: " + str(msg['eval_loss'])[0:8]
                    print(output)


                # Figure out of we should tell this worker to quit.
                parent_model = msg['parent_model']
                worker_id = msg['worker_id']
                num_epochs = msg['num_epochs']
                epoch_id = msg['epoch_id']

                progress = self.update_progress(parent_model, worker_id, num_epochs, epoch_id)
                max_progress = self.max_progress(parent_model)

                if progress < max_progress * 0.75:
                    quit = {}
                    quit['op_code'] = 'quit'
                    self.publish(self.spec['train_channel'] + ':' + worker_id, quit)

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

        datasets = [train_input,train_target,valid_input,valid_target]
        data_json = json.dumps(datasets)
        data_addr = self.api.add_str(data_json)

        spec = {}
        spec['model_addr'] = model_addr
        spec['data_addr'] = data_addr
        spec['batch_size'] = batch_size
        spec['epochs'] = epochs
        spec['log_interval'] = log_interval
        spec['framework'] = framework
        spec['train_channel'] = 'openmined_train_'+str(model_addr)
        return spec
