import torch
import syft

from syft.frameworks.crypten.context import run_party

import crypten.communicator as comm
import crypten


def load(tag: str, src: int, **kwargs):
    if src == comm.get().get_rank():
        worker = syft.local_worker.get_worker_from_rank(src)
        results = worker.search(tag)

        # Make sure there is only one result
        assert len(results) == 1

        result = results[0]
        result = crypten.native_load(f=None, preloaded=result, src=src, **kwargs)

    else:
        result = crypten.native_load(f=None, src=src, **kwargs)

    return result


__all__ = ["run_party", "load"]
