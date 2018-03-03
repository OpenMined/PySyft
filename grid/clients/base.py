from ..workers import base_worker
from .. import channels
from ..lib import strings

from ..services.passively_broadcast_membership import PassivelyBroadcastMembershipService
from ..services.listen_for_openmined_nodes import ListenForOpenMinedNodesService

from colorama import Fore, Back, Style
import json
import time

class BaseClient(base_worker.GridWorker):

    def __init__(self,min_om_nodes=1,known_workers=list(),include_github_known_workers=True):
        super().__init__()
        self.progress = {}

        self.services = {}

        self.services['listen_for_openmined_nodes'] = ListenForOpenMinedNodesService(self,min_om_nodes,include_github_known_workers)
        # self.listen_for_openmined_nodes(min_om_nodes,include_github_known_workers)
        
        self.stats = self.refresh_network_stats()



    def refresh(self,refresh_known_nodes = True, refresh_network_stats=True):
        if(refresh_known_nodes):
            self.services['listen_for_openmined_nodes'].listen_for_openmined_nodes()
        if(refresh_network_stats):
            self.stats = self.refresh_network_stats()

    def get_stats(self,worker_id):

        def ret(msg):
            return json.loads(msg['data'])

        return self.request_response(channel=channels.whoami_listener_callback(worker_id),message=[],response_handler=ret)

    def __len__(self):
        return len(self.get_openmined_nodes())

    # def print_network_stats(self):
    #     for i,n in enumerate(stats):


    def refresh_network_stats(self,print_stats=True):
        om_nodes = self.get_openmined_nodes()
        om_nodes

        stats = list()
        for idx,n in enumerate(self.get_openmined_nodes()):
            start = time.time()
            stat = self.get_stats(n)
            end = time.time()
            stat['ping_time'] = end-start
            stats.append(stat)
            if(print_stats):
                print(self.pretty_print_node(idx,stat))

        self.stats = stats
        return stats

    def pretty_print_gpu(self,gpu):
        return str(gpu['index']) + " : " + gpu['name'] + " : " + str(gpu['memory.used']) + "/" + str(gpu['memory.total'])

    def pretty_print_compute(self,idx,stat):

        wtype = stat['worker_type']
        ncpu = stat['cpu_num_logical_cores']
        cpu_load = stat['cpu_processor_percent_utilization']
        ngpu = len(stat['gpus'])
        dp = stat['disk_percent']
        rp = str(100-stat['cpu_ram_percent_available'])[0:4]
        
        if(ngpu == 0):
            gpus = "[]"
        else:
            gpus = "["
            for g in stat['gpus']:
                gpus += self.pretty_print_gpu(g) + ", "
            gpus = gpus[:-2] + "]"

        ping = str(stat['ping_time']).split(".")
        ping = ping[0] + "." + ping[1][0:2]

        
        return wtype + " - " + str(idx) + " - ID:"+str(stat['id'][-5:])+"  Ping:" + str(ping) + "sec  CPUs:" + str(ncpu) + "  CPU Load:" + str(cpu_load) + "  Disk-util:" + str(dp) + "%" + "  RAM-util:" + str(rp) + "%  GPUs:" + gpus

    def pretty_print_node(self,idx,node):

        if(node['worker_type'] == 'ANCHOR'):
            node['worker_type'] = ' ANCHOR'
        stat_str = self.pretty_print_compute(idx,node)
        if(node['worker_type'] == ' ANCHOR'):
            stat_str = f'{Fore.LIGHTBLACK_EX}' +stat_str+ f'{Style.RESET_ALL}'
        
        return stat_str

    def __getitem__(self,idx):
        return self.stats[idx]['id']

    def __len__(self):
        return len(self.get_openmined_nodes())

