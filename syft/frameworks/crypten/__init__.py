import syft

from syft.frameworks.crypten.model import OnnxModel  # noqa: F401

import crypten.communicator as comm
import crypten

from syft.workers.base import BaseWorker


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
            "CrypTen computation not initiated properly, computation_id doesn't match any rank to "
            "worker_id translation table"
        )
    return syft.local_worker._get_worker_based_on_id(rank_to_worker_id[rank])


def load(tag: str, src: int, **kwargs):
    """TODO: Think of a method to keep the serialized models at the workers that are part of the
    computation in such a way that the worker that started the computation do not know what
    model architecture is used

    if tag.startswith("crypten_model"):
        worker = get_worker_from_rank(src)
        results = worker.search(tag)
        assert len(results) == 1

        model = results[0]
        assert isinstance(model, OnnxModel)

        return utils.onnx_to_crypten(model.serialized_model)
    """

    if src == comm.get().get_rank():
        if CID is None:
            raise RuntimeError("CrypTen computation id is not set.")

        worker = get_worker_from_rank(src)
        results = worker.search(tag)

        # Make sure there is only one result
        assert len(results) == 1

        result = crypten.load_from_party(preloaded=results[0], src=src, **kwargs)

    else:
        result = crypten.load_from_party(preloaded=-1, src=src, **kwargs)

    return result


def load_model(tag: str):
    """
    WARNING: All the workers that are part of the CrypTen computation
    should have the model
    This method should disappear once CrypTen has support to load models
    that are available only at one party (by using the crypten.load function)
    """
    src = comm.get().get_rank()
    worker = get_worker_from_rank(src)
    results = worker.search(tag)

    # Make sure there is only one result
    assert len(results) == 1

    result = results[0].to_crypten()
    return result
