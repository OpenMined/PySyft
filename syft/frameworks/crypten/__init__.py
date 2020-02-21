import torch
import syft as sy
from syft.frameworks.crypten.context import run_party
import crypten.communicator as comm
import crypten


def load(tag: str, src: int, id_worker: int, size: tuple):
    if comm.get().get_rank() == src:
        worker = sy.local_worker.get_worker(id_worker)
        result = worker.search(tag)[0].get()

        # file contains torch.tensor
        if torch.is_tensor(result):
            # Broadcast load type
            load_type = torch.tensor(0, dtype=torch.long)
            comm.get().broadcast(load_type, src=src)

            result = crypten.mpc.MPCTensor(result, src=src)
    else:
        # Receive load type from source party
        load_type = torch.tensor(-1, dtype=torch.long)
        comm.get().broadcast(load_type, src=src)

        # Load in tensor
        if load_type.item() == 0:
            result = crypten.mpc.MPCTensor(torch.empty(size), src=src)
        else:
            raise TypeError("Unrecognized load type on src")
    # TODO: Encrypt modules before returning them

    return result

__all__ = ["run_party", "load", "get_plain_text"]
