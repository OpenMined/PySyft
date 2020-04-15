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


class _WrappedCryptenModel:
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
        return self._model.forward(*args, **kwargs)

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


__all__ = ["run_party", "load", "crypten_to_syft_model"]
