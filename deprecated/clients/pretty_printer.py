from colorama import Fore, Style


class PrettyPrinter:
    def print_gpu(self, gpu):
        return str(gpu['index']) + " : " + gpu['name'] + " : " + str(
            gpu['memory.used']) + "/" + str(gpu['memory.total'])

    def print_compute(self, idx, stat):

        wtype = stat['worker_type']
        ncpu = stat['cpu_num_logical_cores']
        cpu_load = stat['cpu_processor_percent_utilization']
        ngpu = len(stat['gpus'])
        dp = stat['disk_percent']
        rp = str(100 - stat['cpu_ram_percent_available'])[0:4]

        if (ngpu == 0):
            gpus = "[]"
        else:
            gpus = "["
            for g in stat['gpus']:
                gpus += self.print_gpu(g) + ", "
            gpus = gpus[:-2] + "]"

        ping = str(stat['ping_time']).split(".")
        ping = ping[0] + "." + ping[1][0:2]

        if 'name' in stat and stat['name']:
            return wtype + " - " + str(
                idx) + " - NAME:" + stat['name'] + "  Ping:" + str(
                    ping) + "sec  CPUs:" + str(ncpu) + "  CPU Load:" + str(
                        cpu_load) + "  Disk-util:" + str(
                            dp) + "%" + "  RAM-util:" + str(
                                rp) + "%  GPUs:" + gpus
        else:
            return wtype + " - " + str(idx) + " - ID:" + str(
                stat['id'][-5:]) + "  Ping:" + str(ping) + "sec  CPUs:" + str(
                    ncpu
                ) + "  CPU Load:" + str(cpu_load) + "  Disk-util:" + str(
                    dp) + "%" + "  RAM-util:" + str(rp) + "%  GPUs:" + gpus

    def print_node(self, idx, node):

        if (node['worker_type'] == 'ANCHOR'):
            node['worker_type'] = ' ANCHOR'
        stat_str = self.print_compute(idx, node)
        if (node['worker_type'] == ' ANCHOR'):
            stat_str = f'{Fore.LIGHTBLACK_EX}' + stat_str + f'{Style.RESET_ALL}'

        return stat_str
