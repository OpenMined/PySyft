from .. import channels
from .base import BaseService
import json
import torch
from ..lib import utils


class WhoamiService(BaseService):

    # This service just facilitates a worker describing things about itself to the outside world
    # upon request. Note - don't add anything to this service that could be dangerous if made public.

    def __init__(self, worker):
        super().__init__(worker)

        # TODO these below should be listening on self.worker.id but the client
        # does not yet know to ask for info on "computer:IPFS_ADDRESS" yet
        # so just listen on IPFS_ADDRESS
        print(channels.whoami_listener_callback(utils.get_ipfs_id(self.api)))
        self.worker.listen_to_channel(
            channels.whoami_listener_callback(utils.get_ipfs_id(self.api)),
            self.get_stats)

    def get_stats(self, message_and_response_channel):

        msg, response_channel = json.loads(
            message_and_response_channel['data'])

        stats = {}

        import psutil
        stats['worker_type'] = self.worker.node_type
        stats['services_running'] = list(self.worker.services.keys())
        stats['id'] = self.worker.id
        stats['email'] = self.worker.email
        stats['name'] = self.worker.name

        if ('torch_service' in self.worker.services.keys()):
            stats['torch'] = {}
            stats['torch']['objects'] = list(
                self.worker.services['torch_service'].objects.keys())

        stats['cpu_processor_percent_utilization'] = psutil.cpu_percent()
        stats['cpu_num_cores'] = psutil.cpu_count(logical=False)
        stats['cpu_num_logical_cores'] = psutil.cpu_count(logical=True)

        ram = psutil.virtual_memory()

        stats['cpu_ram_total'] = ram.total
        stats['cpu_ram_available'] = ram.available
        stats['cpu_ram_percent_available'] = ram.percent
        stats['cpu_ram_used'] = ram.used
        stats['cpu_ram_free'] = ram.free
        stats['cpu_ram_active'] = ram.active
        stats['cpu_ram_inactive'] = ram.inactive
        # stats['cpu_ram_wired'] = ram.wired

        disk = psutil.disk_usage('/')

        stats['disk_total'] = disk.total
        stats['disk_used'] = disk.used
        stats['disk_free'] = disk.free
        stats['disk_percent'] = disk.percent

        # running this seems to soak up all the GPU memory
        # from tensorflow.python.client import device_lib

        # def get_available_gpus():
        #     local_device_protos = device_lib.list_local_devices()
        #     return [x for x in local_device_protos if x.device_type == 'GPU']

        # def get_available_cpus():
        #     local_device_protos = device_lib.list_local_devices()
        #     return [x for x in local_device_protos if x.device_type == 'CPU']

        # gpus = get_available_gpus()
        # cpus = get_available_cpus()

        # stats['gpus_tf'] = list()

        # for gpu in gpus:
        #   gpu_stats = {}
        #   gpu_stats['memory_limit_tf'] = gpu.memory_limit

        #   desc_list =gpu.physical_device_desc.split(",")
        #   gpu_stats['device_id_tf'] = int(desc_list[0].split(":")[1])
        #   gpu_stats['device_name_tf'] = desc_list[1].split(":")[1].strip()
        #   stats['gpus_tf'].append(gpu_stats)

        stats['gpus'] = list()

        try:
            import gpustat
            for gpu in gpustat.new_query().gpus:
                stats['gpus'].append(gpu.jsonify())
        except:
            ""

        stats['gpus_pytorch'] = list()
        for i in range(torch.cuda.device_count()):
            gpu_stats = {}
            gpu_stats['name'] = torch.cuda.get_device_name(i)
            gpu_stats['cuda_major_verison'], gpu_stats[
                'cuda_minor_verison'] = torch.cuda.get_device_capability(i)
            stats['gpus_pytorch'].append(gpu_stats)

        self.worker.publish(
            channel=response_channel, message=json.dumps(stats))
