import syft

from syft.frameworks.crypten import utils

import crypten.communicator as comm
import crypten

from syft.workers.base import BaseWorker
from syft.generic.pointers.object_pointer import ObjectPointer


RANK_TO_WORKER_ID = {
    # Contains translation dictionaries for every computation.
    # cid (computation id): {rank_to_worker_id dictionary for a specific computation}
}
CID = None


def get_worker_from_rank(rank: int, cid: int = None) -> BaseWorker:
    """Find the worker running CrypTen party with specific rank in a certain computation.

    Args:
        rank: rank of the CrypTen party.
        cid: CrypTen computation id.

    Returns:
        BaseWorker corresponding to cid and rank.
    """
    if cid is None:
        if CID is None:
            # Neither CID have been set appropriately nor cid have been passed
            raise ValueError("cid must be set.")
        cid = CID

    rank_to_worker_id = RANK_TO_WORKER_ID.get(cid, None)
    if rank_to_worker_id is None:
        raise RuntimeError(
            "CrypTen computation not initiated properly, computation_id doesn't match any rank to"
            "worker_id translation table"
        )
    return syft.local_worker._get_worker_based_on_id(rank_to_worker_id[rank])


def load(tag: str, src: int, **kwargs):
    if tag.startswith("crypten_model"):
        worker = get_worker_from_rank(src)
        results = worker.search(tag)
        assert len(results) == 1

        result = results[0]

        if isinstance(result, ObjectPointer):
            model = result.clone().get()
        else:
            model = result

        return utils.onnx_to_crypten(model.serialized_model)

    if src == comm.get().get_rank():
        if CID is None:
            raise RuntimeError("CrypTen computation id is not set.")

        worker = get_worker_from_rank(src)
        results = worker.search(tag)
        assert len(results) == 1

        result = crypten.load_from_party(preloaded=results[0], src=src, **kwargs)

    else:
        result = crypten.load_from_party(preloaded=-1, src=src, **kwargs)

    return result
