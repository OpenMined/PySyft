from grid.lib import utils
import random
import base64
import json
import numpy as np
import sys
import asyncio


class PubSub(object):
    def __init__(self, ipfs_addr='127.0.0.1', port=5001):
        self.api = utils.get_ipfs_api()
        self.encoded_id = self.get_encoded_id()
        self.id = self.api.config_show()['Identity']['PeerID']

    def serialize_numpy(self, tensor):
        # nested lists with same data, indices
        return json.dumps(tensor.tolist())

    def deserialize_numpy(self, json_array):
        return np.array(json.loads(json_array)).astype('float')

    def publish(self, channel, dict_message):
        self.api.pubsub_pub(topic=channel, payload=json.dumps(dict_message))


    def listen_to_channel(self, channel, handle_message, init_function=None, ignore_from_self=False):
        first_proc = True
        new_models = self.api.pubsub_sub(topic=channel, stream=True)

        for m in new_models:
            if init_function is not None and first_proc:
                init_function()
                first_proc = False
            message = self.decode_message(m)

            if not ignore_from_self or message['from'] != self.encoded_id:
                if(message is not None):
                    out = handle_message(message)
                    if(out is not None):
                        return out
            else:
                print("ignored message from self")

    def get_encoded_id(self):
        """Currently a workaround because we can't figure out how to decode the 'from'
        side of messages sent across the wire. However, we can check to see if two messages
        are equal. Thus, by sending a random message to ourselves we can figure out what
        our own encoded id is. TODO: figure out how to decode it."""

        rand_channel = random.randint(0, 1000000)
        try:
            temp_channel = self.api.pubsub_sub(topic=rand_channel, stream=True)
        except:
            print(f'\n{Fore.RED}ERROR: {Style.RESET_ALL}could not connect to IPFS PUBSUB.  Did you run the daemon with {Fore.GREEN}--enable-pubsub-experiment{Style.RESET_ALL} ?')
            sys.exit()

        secret = random.randint(0, 1000000)
        self.api.pubsub_pub(topic=rand_channel, payload="id:" + str(secret))

        for encoded in temp_channel:

            # decode message
            decoded = self.decode_message(encoded)

            if(decoded is not None):
                if(str(decoded['data'].split(":")[-1]) == str(secret)):
                    return str(decoded['from'])

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
            f.close()
        return model_bin

    def deserialize_torch_model(self, model_bin, model_class, **kwargs):
        """
        model_class is needed since PyTorch uses pickle for serialization
            see https://discuss.pytorch.org/t/loading-pytorch-model-without-a-code/12469/2 for details
        kwargs are the arguments needed to instantiate the model from model_class
        """
        with open('temp_model2.pth.tar', 'wb') as g:
            g.write(model_bin)
            g.close()
        state = torch.load()
        model = model_class(**state['kwargs'])
        model.load_state_dict(state['state_dict'])
        return model
