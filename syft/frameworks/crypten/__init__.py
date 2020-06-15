import syft

import crypten.communicator as comm
import crypten

from syft.workers.base import BaseWorker


RANK_TO_WORKER_ID = {}
CID = None


def get_worker_from_rank(cid: int, rank: int) -> BaseWorker:
    """Find the worker running CrypTen party with specific rank in a certain computation.

    Args:
        cid: CrypTen computation id.
        rank: rank of the CrypTen party.

    Returns:
        BaseWorker corresponding to cid and rank.
    """
    rank_to_worker_id = RANK_TO_WORKER_ID.get(cid, None)
    if rank_to_worker_id is None:
        raise RuntimeError(
            "CrypTen computation not initiated properly, computation_id doesn't match any rank to"
            "worker_id transaltion table"
        )
    return syft.local_worker._get_worker_based_on_id(rank_to_worker_id[rank])


def load(tag: str, src: int, **kwargs):
    if src == comm.get().get_rank():
        if CID is None:
            raise RuntimeError("CrypTen computation id is not set.")

        worker = get_worker_from_rank(CID, src)
        results = worker.search(tag)

        # Make sure there is only one result
        assert len(results) == 1

        result = results[0]
        result = crypten.load_from_party(preloaded=result, src=src, **kwargs)

    else:
        result = crypten.load_from_party(preloaded=-1, src=src, **kwargs)

    return result
