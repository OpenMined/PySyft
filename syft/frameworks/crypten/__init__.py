import torch
import syft

from syft.frameworks.crypten.context import toy_func, run_party
from syft.exceptions import WorkerNotFoundException

import crypten.communicator as comm
import crypten


def load(tag: str, src: int):
    worker_id = syft.local_worker.get_worker_id_from_rank(src)
    assert worker_id != -1

    try:
        worker = syft.local_worker.get_worker(worker_id, fail_hard=True)
        results = worker.search(tag)

        # Make sure there is only one result
        assert len(results) == 1

        result = results[0].get()

        if torch.is_tensor(result):

            # Broadcast load type
            load_type = torch.tensor(0, dtype=torch.long)
            comm.get().broadcast(load_type, src=src)

            # Broadcast size to other parties.
            dim = torch.tensor(result.dim(), dtype=torch.long)
            size = torch.tensor(result.size(), dtype=torch.long)

            comm.get().broadcast(dim, src=src)
            comm.get().broadcast(size, src=src)
            result = crypten.mpc.MPCTensor(result, src=src)
        else:
            raise TypeError("Unrecognized load type on src")

    except WorkerNotFoundException:
        # Receive load type from source party
        load_type = torch.tensor(-1, dtype=torch.long)
        comm.get().broadcast(load_type, src=src)

        # Load in tensor
        if load_type.item() == 0:
            # Receive size from source party
            dim = torch.empty(size=(), dtype=torch.long)
            comm.get().broadcast(dim, src=src)
            size = torch.empty(size=(dim.item(),), dtype=torch.long)
            comm.get().broadcast(size, src=src)
            result = crypten.mpc.MPCTensor(torch.empty(size=tuple(size.tolist())), src=src)
        else:
            raise TypeError("Unrecognized load type on src")

    return result


__all__ = ["toy_func", "run_party", "load"]
