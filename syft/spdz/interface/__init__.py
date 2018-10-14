from syft.spdz.interface import base_interface
from syft.spdz.interface import distributed_interface
from syft.spdz.interface import grid_client_interface
from syft.spdz.interface import grid_worker_interface

s = str(base_interface)
s += str(distributed_interface)
s += str(grid_client_interface)
s += str(grid_worker_interface)
