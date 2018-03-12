from ..workers import base_worker
from .. import channels
from grid.lib import utils
from .pretty_printer import PrettyPrinter

from ..services.listen_for_openmined_nodes import ListenForOpenMinedNodesService

from threading import Thread
from colorama import Fore, Style
import json
import time
import ipywidgets as widgets


class BaseClient(base_worker.GridWorker):
    def __init__(self,
                 min_om_nodes=1,
                 known_workers=list(),
                 include_github_known_workers=True,
                 include_self_discovery=True,
                 verbose=True):
        super().__init__('client')
        self.progress = {}
        self.services = {}

        self.services[
            'listen_for_openmined_nodes'] = ListenForOpenMinedNodesService(
                self,
                min_om_nodes,
                known_workers,
                include_github_known_workers)

        self.stats = list()

        self.pretty_printer = PrettyPrinter()

        self.include_self_discovery = include_self_discovery

        def ping_known_then_refresh():

            for w in self.services['listen_for_openmined_nodes'].known_workers:
                try:
                    self.stats.append(self.get_stats(w.split("/")[-1]))
                except:
                    ""
            self.refresh_network_stats(print_stats=verbose)

        t1 = Thread(target=ping_known_then_refresh, args=[])
        t1.start()

    def refresh(self, refresh_known_nodes=True, refresh_network_stats=True):
        if (refresh_known_nodes):
            self.services[
                'listen_for_openmined_nodes'].listen_for_openmined_nodes()
        if (refresh_network_stats):
            self.stats = self.refresh_network_stats()

    def get_stats(self, worker_id, timeout=10):
        def ret(msg):
            return json.loads(msg['data'])

        return self.request_response(
            channel=channels.whoami_listener_callback(worker_id),
            message=[],
            response_handler=ret,
            timeout=10)

    def print_network_stats(self):
        for i, n in enumerate(self.stats):
            try:
                print(self.pretty_printer.print_node(i, n))
            except:
                "was probably being written to asyncronously"

    def refresh_network_stats(self, print_stats=True):
        om_nodes = self.get_openmined_nodes()
        if self.include_self_discovery:
            om_nodes.append(utils.get_ipfs_id(self.api))

        if (self.stats is not None):
            # try to preserve the order to that g[idx] stays the same
            existing_stats = set()
            for s in self.stats:
                existing_stats.add(s['id'])

            new_om_nodes = set()

            for om_node in om_nodes:
                if (om_node not in existing_stats):
                    new_om_nodes.add(om_node)

            self.old_stats = self.stats
            self.stats = list()
            for idx, old_stat in enumerate(self.old_stats):

                if (old_stat['id'] in om_nodes):
                    start = time.time()

                    try:
                        stat = self.get_stats(old_stat['id'])
                    except TimeoutError:
                        if (print_stats):
                            print(f'{Fore.LIGHTBLACK_EX}' + "NODE    - " +
                                  str(idx) + " - - timeout - - " +
                                  str(old_stat['id']) + f'{Style.RESET_ALL}')

                        continue

                    end = time.time()
                    stat['ping_time'] = end - start
                    stat['status'] = 'ONLINE'
                else:
                    stat = old_stat
                    stat['status'] = 'OFFLINE'

                self.stats.append(stat)
                if (print_stats):
                    print(
                        self.pretty_printer.print_node(
                            len(self.stats) - 1, stat))

            for idx_, id in enumerate(new_om_nodes):
                idx = len(self.stats)
                start = time.time()

                try:
                    stat = self.get_stats(id)
                except TimeoutError:
                    if (print_stats):
                        print(f'{Fore.LIGHTBLACK_EX}' + "NODE    - " +
                              str(idx) + " - - timeout - - " + str(id) +
                              f'{Style.RESET_ALL}')

                    continue

                end = time.time()
                stat['ping_time'] = end - start
                stat['status'] = 'ONLINE'
                self.stats.append(stat)
                if (print_stats):
                    print(
                        self.pretty_printer.print_node(
                            len(self.stats) - 1, stat))

        else:
            self.old_stats = self.stats
            self.stats = list()
            for idx, id in enumerate(om_nodes):
                start = time.time()
                try:
                    stat = self.get_stats(id)
                except TimeoutError:
                    if (print_stats):
                        print(f'{Fore.LIGHTBLACK_EX}' + "NODE    - " +
                              str(idx) + " - - timeout - - " + str(id) +
                              f'{Style.RESET_ALL}')
                    continue
                end = time.time()
                stat['ping_time'] = end - start
                self.stats.append(stat)
                if (print_stats):
                    print(self.pretty_printer.print_node(idx, stat))

        return self.stats

    def __getitem__(self, idx):
        return self.stats[idx]['id']

    def __len__(self):
        return len(self.get_openmined_nodes())

    """
    Grid Tree Implementation

    Methods for Grid tree down here
    """

    def found_task(self, message):
        tasks = json.loads(message['data'])
        for task in tasks:
            # utils.store_task(task['name'], task['address'])
            name = task['name']
            addr = task['address']

            hbox = widgets.HBox([widgets.Label(name), widgets.Label(addr)])
            self.all_tasks.children += (hbox, )

    def find_tasks(self):
        self.publish(channels.list_tasks, "None")
        self.all_tasks = widgets.VBox([
            widgets.HBox(
                [widgets.Label('TASK NAME'),
                 widgets.Label('ADDRESS')])
        ])
        self.listen_to_channel(
            channels.list_tasks_callback(self.id), self.found_task)

        return self.all_tasks

    def add_task(self, name, data_dir=None, adapter=None):
        if data_dir is None and adapter is None:
            print(
                f'{Fore.RED}data_dir and adapter can not both be None{Style.RESET_ALL}'
            )
            return

        task_data = {'name': name, 'creator': self.id}

        if data_dir is not None:
            task_data['data_dir'] = data_dir
        if adapter is not None:
            with open(adapter, 'rb') as f:
                adapter_bin = f.read()
                f.close()
            adapter_addr = self.api.add_bytes(adapter_bin)
            task_data['adapter'] = adapter_addr

        addr = self.api.add_json(task_data)
        utils.store_task(name, addr)

        data = json.dumps([{'name': name, 'address': addr}])
        self.publish('openmined:add_task', data)
