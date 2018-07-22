from .. import channels
import json
import time
import sys
from colorama import Fore, Style
from .base import BaseService
import requests


class ListenForOpenMinedNodesService(BaseService):

    # Blocking until this node has found at least one other OpenMined node
    # This functionality queries https://github.com/OpenMined/BootstrapNodes for Anchor nodes
    # then asks those nodes for which other OpenMined nodes they know about on the network.

    def __init__(self,
                 worker,
                 min_om_nodes=1,
                 known_workers=list(),
                 include_github_known_workers=True):
        super().__init__(worker)

        self.listen_for_openmined_nodes(
            min_om_nodes=min_om_nodes,
            known_workers=known_workers,
            include_github_known_workers=include_github_known_workers)

    def download_github_known_workers(self):
        workers = requests.get(
            'https://github.com/OpenMined/BootstrapNodes/raw/master/known_workers'
        ).text.split("\n")
        for w in workers:
            if ('p2p-circuit' in w):
                self.known_workers.append(w)

    def connect_known_workers(self):
        # if there are known workers - connect with them directly
        if (len(self.known_workers) > 0):
            print(
                f'\n{Fore.BLUE}UPDATE: {Style.RESET_ALL}Querying known workers...'
            )
            for worker in self.known_workers:
                try:
                    sys.stdout.write('\tWORKER: ' + str(worker) + '...')
                    self.worker.api.swarm_connect(worker)
                    sys.stdout.write(
                        f'{Fore.GREEN}SUCCESS!!!{Style.RESET_ALL}\n')
                except:
                    sys.stdout.write(f'{Fore.RED}FAIL!!!{Style.RESET_ALL}\n')
                    ""

    def listen_for_openmined_nodes(self,
                                   min_om_nodes=1,
                                   known_workers=list(),
                                   include_github_known_workers=True):
        self.known_workers = known_workers

        # pull known workers from trusted github source (OpenMined's repo)
        if (include_github_known_workers):
            self.download_github_known_workers()

        # remove duplicates
        self.known_workers = list(set(self.known_workers))

        self.connect_known_workers()

        def load_proposed_workers(message):

            failed = list()

            for worker in json.loads(message['data']):
                addr = '/p2p-circuit/ipfs/' + worker
                try:
                    self.worker.api.swarm_connect(addr)
                except:
                    failed.append(addr)

            time.sleep(5)

            still_failed = list()
            for addr in failed:
                try:
                    self.worker.api.swarm_connect(addr)
                except:
                    still_failed.append(addr)

            time.sleep(10)

            for addr in still_failed:
                try:
                    self.worker.api.swarm_connect(addr)
                except:
                    "give up"

            return message

        self.worker.listen_to_channel(
            channels.list_workers_callback(self.worker.id),
            load_proposed_workers)
        self.worker.publish(
            channel='openmined:list_workers', message={
                'key': 'value'
            })

        num_nodes_om = 0

        print("\n")

        logi = 0
        while (True):

            time.sleep(0.25)
            num_nodes_total = len(self.worker.get_nodes())
            num_nodes_om = len(self.worker.get_openmined_nodes())

            sys.stdout.write(
                f'\r{Fore.BLUE}UPDATE: {Style.RESET_ALL}Searching for IPFS nodes - '
                + str(num_nodes_total) + ' found overall - ' +
                str(num_nodes_om) + ' are OpenMined workers' +
                ("." * (logi % 10)) + (' ' * 10))

            logi += 1

            if (num_nodes_om >= min_om_nodes):
                break

            if (logi % 100 == 99):
                self.worker.publish(
                    channel='openmined:list_workers', message={
                        'key': 'value'
                    })
                print(">")

        print(f'\n\n{Fore.GREEN}SUCCESS: {Style.RESET_ALL}Found ' +
              str(num_nodes_om) + ' OpenMined nodes!!!\n')
