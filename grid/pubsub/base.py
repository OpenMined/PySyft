from grid.lib import utils
from grid.pubsub import channels
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

    def get_openmined_nodes(self):
        nodes = self.api.pubsub_peers('openmined')['Strings']
        if(nodes is not None):
            return nodes
        else:
            return []

    def get_nodes(self):
        nodes =  self.api.pubsub_peers()['Strings']
        if(nodes is not None):
            return nodes
        else:
            return []


    def listen_for_openmined_nodes(self, min_om_nodes = 1):
        
        def load_proposed_workers(message):

            failed = list()

            for worker in json.loads(message['data']):
                addr = '/p2p-circuit/ipfs/'+worker
                try:
                    self.api.swarm_connect(addr)
                except:
                    failed.append(addr)

            time.sleep(5)

            still_failed = list()
            for addr in failed:
                try:
                    self.api.swarm_connect(addr)
                except:
                    still_failed.append(addr)

            time.sleep(10)

            for addr in still_failed:
                try:
                    self.api.swarm_connect(addr)
                except:
                    "give up"

            return message

        self.listen_to_channel(channels.list_workers_callback(self.id),load_proposed_workers)
        self.publish(channel='openmined:list_workers',message={'key':'value'})

        num_nodes_om = 0

        print("")

        logi = 0
        while(True):
            
            time.sleep(0.25)
            num_nodes_total = len(self.get_nodes())
            num_nodes_om = len(self.get_openmined_nodes())

            sys.stdout.write(f'\r{Fore.BLUE}UPDATE: {Style.RESET_ALL}Searching for IPFS nodes - '+str(num_nodes_total)+' found overall - '+str(num_nodes_om)+' are OpenMined workers' + ("." * (logi%10)) + (' ' * 10))
            
            logi += 1

            if(num_nodes_om >= min_om_nodes):
                break

            if(logi % 100 == 99):
                self.publish(channel='openmined:list_workers',message={'key':'value'})
                print(">")

                
                
                
                

        print(f'\n\n{Fore.GREEN}SUCCESS: {Style.RESET_ALL}Found '+str(num_nodes_om)+' OpenMined nodes!!!\n')    

    def serialize_numpy(self, tensor):
        # nested lists with same data, indices
        return json.dumps(tensor.tolist())

    def deserialize_numpy(self, json_array):
        return np.array(json.loads(json_array)).astype('float')

    def publish(self, channel, message):
        if isinstance(message, dict) or isinstance(message, list):
            self.api.pubsub_pub(topic=channel, payload=json.dumps(message))
        else:
            self.api.pubsub_pub(topic=channel, payload=message)

    def listen_to_channel_sync(self, *args):
        """
        Synchronous version of listen_to_channel
        """

        self.listen_to_channel_impl(*args)

    def listen_to_channel(self, *args):
        """
        Listens for IPFS pubsub sub messages asynchronously.

        This function will create the listener and call back your handler
        function on a new thread.
        """
        t1 = Thread(target=self.listen_to_channel_impl, args=args)
        t1.start()

    def listen_to_channel_impl(self, channel, handle_message,
                               init_function=None, ignore_from_self=False):
        """
        Do not call directly.  Use listen_to_channel or listen_to_channel_sync instead.
        """

        first_proc = True

        if channel not in self.subscribed_list:
            print(f"SUBSCRIBING TO {channel}")
            new_messages = self.api.pubsub_sub(topic=channel, stream=True)
            self.subscribed_list.append(channel)
        else:
            print(f"ALREADY SUBSCRIBED TO {channel}")
            return

        for m in new_messages:
            if init_function is not None and first_proc:
                init_function()
                first_proc = False

            message = self.decode_message(m)
            if message is not None:
                fr = base58.encode(message['from'])

                if not ignore_from_self or fr != self.id:
                    out = handle_message(message)
                    if(out is not None):
                        return out
                else:
                    print("ignored message from self")

                # if(message is not None):
                #    if handle_message is not None:
                #        out = handle_message(message)
                #        if (out is not None):
                #            return out
                #        else:
                #            return message
                #    else:
                #        return message

    def decode_message(self, encoded):
        if('from' in encoded):
            decoded = {}
            decoded['from'] = base64.standard_b64decode(encoded['from'])
            decoded['data'] = base64.standard_b64decode(
                encoded['data']).decode('ascii')
            decoded['seqno'] = base64.standard_b64decode(encoded['seqno'])
            decoded['topicIDs'] = encoded['topicIDs']
            decoded['encoded'] = encoded
            return decoded
        else:
            return None

    def serialize_torch_model(self, model, **kwargs):
        """
        kwargs are the arguments needed to instantiate the model
        """
        state = {'state_dict': model.state_dict(), 'kwargs': kwargs}
        torch.save(state, 'temp_model.pth.tar')
        with open('temp_model.pth.tar', 'rb') as f:
            model_bin = f.read()
        return model_bin

    def deserialize_torch_model(self, model_bin, model_class, **kwargs):
        """
        model_class is needed since PyTorch uses pickle for serialization
            see https://discuss.pytorch.org/t/loading-pytorch-model-without-a-code/12469/2 for details
        kwargs are the arguments needed to instantiate the model from model_class
        """
        with open('temp_model2.pth.tar', 'wb') as g:
            g.write(model_bin)
        state = torch.load()
        model = model_class(**state['kwargs'])
        model.load_state_dict(state['state_dict'])
        return model

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

        model_addr = utils.keras2ipfs(model)

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
