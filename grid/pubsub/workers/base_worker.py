from .. import base

class GridWorker(base.PubSub):

    def __init__(self):
        super().__init__('worker')

        # LAUNCH PROCESSES - these are non-blocking and run on their own threads

        # Blocking until this node has found at least one other OpenMined node
        # This functionality queries https://github.com/OpenMined/BootstrapNodes for Anchor nodes
        # then asks those nodes for which other OpenMined nodes they know about on the network.
        self.listen_for_openmined_nodes(1)

        # this process serves the purpose of helping other nodes find out about nodes on the network.
        # if someone queries the "list_worker" channel - it'll send a message directly to the querying node
        # with a list of the OpenMined nodes of which it is aware.
        self.listen_to_channel(channels.list_workers,self.list_workers)

        # This process listens for models that it can train.
        self.listen_to_channel(channels.openmined, self.fit_worker)
