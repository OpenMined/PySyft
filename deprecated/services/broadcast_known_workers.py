from .. import channels
from .base import BaseService
from bitcoin import base58
import json


class BroadcastKnownWorkersService(BaseService):

    # this process serves the purpose of helping other nodes find out about nodes on the network.
    # if someone queries the "list_worker" channel - it'll send a message directly to the querying node
    # with a list of the OpenMined nodes of which it is aware.

    def __init__(self, worker):
        super().__init__(worker)

        self.worker.listen_to_channel(channels.list_workers,
                                      self.reply_with_list_of_known_workers)

    def reply_with_list_of_known_workers(self, message):

        fr = base58.encode(message['from'])

        addr = '/p2p-circuit/ipfs/' + fr

        # print("sending list of known workers to " + addr)

        # First: adding worker that made request to my list of known workers (just in case i don't have it)
        try:
            self.api.swarm_connect(addr)
        except:
            ""
            # print("Failed to reconnect in the opposite direciton to:" + addr)

        # fetch list of openmined workers i know about
        workers = self.worker.get_openmined_nodes()

        # encode list of openmined workers i know about
        workers_json = json.dumps(workers)

        # get callback channel for sending messages directly to whomever requested the list of om workers i know about
        callback_channel = channels.list_workers_callback(fr)

        # print(f'Sending List: {callback_channel} {workers_json}')

        # send encoded list of openmined workers i know about to the individual who requested it
        self.worker.publish(callback_channel, workers_json)
