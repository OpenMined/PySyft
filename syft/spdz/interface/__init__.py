from syft.spdz.interface import (
    base_interface,
    distributed_interface,
    grid_client_interface,
    grid_worker_interface,
)

s = str(base_interface)
s += str(distributed_interface)
s += str(grid_client_interface)
s += str(grid_worker_interface)
