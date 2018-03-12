from ..services.broadcast_known_workers import BroadcastKnownWorkersService
from ..services.whoami import WhoamiService
from ..lib import utils
from threading import Thread
import json
from bitcoin import base58
import base64
import time
import random


class GridWorker():
    def __init__(self, node_type):
        self.node_type = node_type
        self.api = utils.get_ipfs_api(self.node_type)
        self.id = utils.get_id(self.node_type, self.api)

        # load email and name
        whoami = utils.load_whoami()
        if whoami:
            self.email = whoami['email']
            self.name = whoami['name']
        else:
            self.email = input('Enter your email for payment: ')
            self.name = input('Enter an easy name to remember you by: ')

            whoami = {'email': self.email, 'name': self.name}

            utils.store_whoami(whoami)

        self.subscribed_list = []

        # LAUNCH SERVICES - these are non-blocking and run on their own threads

        # all service objects will live in this dictionary

        self.services = {}

        if node_type != 'client':
            # this service serves the purpose of helping other nodes find out
            # about nodes on the network.
            # if someone queries the "list_worker" channel - it'll send a
            # message directly to the querying node with a list of the
            # OpenMined nodes of which it is aware.
            self.services[
                'broadcast_known_workers'] = BroadcastKnownWorkersService(self)

            # WHOMAI
            self.services['whoami_service'] = WhoamiService(self)

    def get_openmined_nodes(self):
        """
        This method returns the list of known openmined workers on the newtork.
        Note - not all workers are necessarily "compute" workers.
        Some may only be anchors and will ignore any jobs you send them.
        """

        nodes = self.api.pubsub_peers('openmined')['Strings']
        if (nodes is not None):
            return nodes
        else:
            return []

    def get_nodes(self):
        nodes = self.api.pubsub_peers()['Strings']
        if (nodes is not None):
            return nodes
        else:
            return []

    def publish(self, channel, message):
        """
        This method sends a message over an IPFS channel. The number of people
        who receive it is purely based on the number of people who happen
        to be listening.
        """

        if isinstance(message, dict) or isinstance(message, list):
            self.api.pubsub_pub(topic=channel, payload=json.dumps(message))
        else:
            self.api.pubsub_pub(topic=channel, payload=message)

    def request_response(self, channel, message, response_handler, timeout=10):
        """
        This method makes a request over a channel to a specific node and
        will hang until it receives a response from that node. Note that
        the channel used for the response is random.
        """

        random_channel = self.id + "_" + str(random.randint(0, 1e10))

        def timeout_message(seconds):
            time.sleep(int(seconds))
            self.publish(
                channel=random_channel,
                message=["timeout after " + str(seconds) + " seconds"])

        def send():
            self.publish(channel=channel, message=[message, random_channel])
            t1 = Thread(target=timeout_message, args={timeout})
            t1.start()

        response = self.listen_to_channel_sync(random_channel,
                                               response_handler, send)

        if (len(response) == 1):
            if ('timeout' in response[0]):
                raise TimeoutError(response[0])
        return response

    def listen_to_channel_sync(self, *args):
        """
        Synchronous version of listen_to_channel
        """

        return self.listen_to_channel_impl(*args)

    def listen_to_channel(self, *args):
        """
        Listens for IPFS pubsub sub messages asynchronously.

        This function will create the listener and call back your handler
        function on a new thread.
        """
        t1 = Thread(target=self.listen_to_channel_impl, args=args)
        t1.start()

    def listen_to_channel_impl(self,
                               channel,
                               handle_message,
                               init_function=None,
                               ignore_from_self=False):
        """
        Do not call directly.  Use listen_to_channel or listen_to_channel_sync instead.
        """

        first_proc = True

        if channel not in self.subscribed_list:
            new_messages = self.api.pubsub_sub(topic=channel, stream=True)
            self.subscribed_list.append(channel)

        else:
            return

        # new_messages is a generator which will keep yield new messages until
        # you return from the loop. If you do return from the loop, we will no
        # longer be subscribed.
        for m in new_messages:
            if init_function is not None and first_proc:
                init_function()
                first_proc = False

            message = self.decode_message(m)
            if message is not None:
                fr = base58.encode(message['from'])
                if not ignore_from_self or fr != self.id:
                    out = handle_message(message)
                    if out is not None:
                        return out
                else:
                    print('ignore mssage from self')

    def decode_message(self, encoded):
        if ('from' in encoded):
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
