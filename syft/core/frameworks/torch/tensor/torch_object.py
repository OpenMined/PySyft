import syft as sy
from syft.mpc import spdz
from syft.core.frameworks.torch import torch_utils
from syft.core.frameworks.torch import _GeneralizedPointerTensor
from syft.core.frameworks.torch import _MPCTensor


class _TorchObject(object):
    """
    This tensor is simply a more convenient way to add custom
    functions to all Torch tensor types, including Torch Variable.
    Note that it is the parent class of the two following classes:
    _TorchTensor and a_TorchVariable
    """

    __module__ = 'syft'

    def share(self, bob, alice):
        x_enc = spdz.encode(self)
        x_alice, x_bob = spdz.share(x_enc)
        x_alice.send(alice)
        x_bob.send(bob)
        x_pointer_tensor_dict = {alice: x_alice.child, bob: x_bob.child}
        x_gp = _GeneralizedPointerTensor(x_pointer_tensor_dict).on(self)
        x_mpc = _MPCTensor(x_gp)
        return x_mpc

    def set_id(self, new_id):
        self.child.set_id(new_id)
        return self

    def __str__(self):
        return self.native___str__()

    def __repr__(self):

        if torch_utils.is_tensor(self) and hasattr(self, 'child') and not isinstance(self.child, (
                sy._LocalTensor, sy._PointerTensor)):
            x_ = type(self)()
            x_.native_set_(self)
            return "[Head of chain]\n" + x_.native___repr__()

        if torch_utils.is_variable(self) and hasattr(self, 'child') and not isinstance(self.child, (
                sy._LocalTensor, sy._PointerTensor)):
            x_ = type(self)(self.data)
            x_.native_set_(self)
            return "[Head of chain]\n" + x_.native___repr__()

        return self.native___repr__()

    def create_pointer(self, register=False, location=None, ptr_id=None):

        return self.child.create_pointer(parent=self, register=register, location=location,
                                         ptr_id=ptr_id).wrap()

    def move(self, worker, new_id=None):
        """
        Give the end leaf of the chain to worker,
        just like if the last elmt was send its child
        to worker
        self->alice->obj [worker] => self->alice->worker->obj
        """
        raise NotImplementedError('Move is not supported anymore.')
        if isinstance(worker, (int, str)):
            worker = self.owner.get_worker(worker)

        if new_id is None:
            new_id = random.randint(0, 10e10)

        if isinstance(self.child, sy._PointerTensor):
            pointer = self.child
        else:
            pointer = None

        if pointer is None:
            return self.send(worker, new_id)

        command, _ = pointer.compile_command('move',
                                             (worker.id, new_id),
                                             {},
                                             True)

        response = pointer.owner.send_torch_command(recipient=pointer.location,
                                                    message=command)
        return self
