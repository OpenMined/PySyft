import torch
import syft
from syft.frameworks.crypten.context import toy_func, run_party
import crypten.communicator as comm
import crypten


def load(tag: str, src: int, id_worker: int):
    if comm.get().get_rank() == src:
        worker = syft.local_worker.get_worker(id_worker)
        result = worker.search(tag)[0].get()

        # file contains torch.tensor
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
    # TODO: Encrypt modules before returning them

    return result


class _WrappedCryptenModel():
    """Wrap a crypten model to offer the same API as syft does for
    torch modules.
    """

    # TODO: forward any other function call to the underlying crypten model
    # TODO: create a copy of the crypten_model?

    def __init__(self, crypten_model):
        if crypten_model.encrypted:
            raise TypeError("Crypten model must be unencrypted.")
        self._model = crypten_model

    def parameters(self):
        for p in self._model.parameters():
            yield p

    def share(self, *args, **kwargs):
        for p in self.parameters():
            p.share_(*args, **kwargs)
        return self

    def forward(self, *args, **kwargs):
        return  self._model.forward(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def fix_prec(self, *args, **kwargs):
        for p in self.parameters():
            p.fix_precision_(*args, **kwargs)
        return self

    def float_prec(self):
        for p in self.parameters():
            p.float_prec()
        return self

    def send(self, *dest, **kwargs):
        for p in self.parameters():
            p.send_(*dest, **kwargs)
        return self

    def move(self, dest):
        for p in self.parameters():
            p.move(dest)
        return self

    def get(self):
        for p in self.parameters():
            p.get_()
        return self

    def copy(self):
        pass

    @property
    def owner(self):
        for p in self.parameters():
            return p.owner

    @property
    def location(self):
        try:
            for p in self.parameters():
                return p.location
        except AttributeError:
            raise AttributeError(
                "Module has no attribute location, did you already send it to some location?"
            )


def crypten_to_syft_model(crypten_model):
    """Transform a crypten model to an object that have the same
    API as syft does for torch modules.

    Args:
        crypten_model: the crypten.nn.Module object to be transformed.
    """
    return _WrappedCryptenModel(crypten_model)


__all__ = ["toy_func", "run_party", "load", "crypten_to_syft_model"]
