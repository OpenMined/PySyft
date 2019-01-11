import pytest
import torch
import torch.nn.functional as F
import random

import syft
from syft.exceptions import RemoteTensorFoundError
from syft.frameworks.torch.tensors import PointerTensor


class TestHook(object):
    def setUp(self):
        hook = syft.TorchHook(torch, verbose=True)

        self.me = hook.local_worker
        self.me.is_client_worker = True

        instance_id = str(int(10e10 * random.random()))
        bob = syft.VirtualWorker(id=f"bob{instance_id}", hook=hook, is_client_worker=False)
        alice = syft.VirtualWorker(id=f"alice{instance_id}", hook=hook, is_client_worker=False)
        james = syft.VirtualWorker(id=f"james{instance_id}", hook=hook, is_client_worker=False)

        bob.add_workers([alice, james])
        alice.add_workers([bob, james])
        james.add_workers([bob, alice])

        self.hook = hook

        self.bob = bob
        self.alice = alice
        self.james = james

    def test___init__(self):
        self.setUp()
        assert torch.torch_hooked
        assert self.hook.torch.__version__ == torch.__version__

    def test_torch_attributes(self):
        with pytest.raises(RuntimeError):
            syft.torch._command_guard("false_command", "torch_modules")

        assert syft.torch._is_command_valid_guard("torch.add", "torch_modules")
        assert not syft.torch._is_command_valid_guard("false_command", "torch_modules")

        syft.torch._command_guard("torch.add", "torch_modules", get_native=False)

    def test_worker_registration(self):
        self.setUp()
        boris = syft.VirtualWorker(id="boris", hook=self.hook, is_client_worker=False)

        self.me.add_workers([boris])

        worker = self.me.get_worker(boris)

        assert boris == worker

    def test_pointer_found_exception(self):
        self.setUp()
        ptr_id = int(10e10 * random.random())
        pointer = PointerTensor(id=ptr_id, location=self.alice, owner=self.me)
        try:
            raise RemoteTensorFoundError(pointer)
        except RemoteTensorFoundError as err:
            err_pointer = err.pointer
            assert isinstance(err_pointer, PointerTensor)
            assert err_pointer.id == ptr_id

    @pytest.mark.parametrize("attr", ["abs"])
    def test_get_pointer_unary_method(self, attr):
        self.setUp()
        x = torch.Tensor([1, 2, 3])
        native_method = getattr(x, f"native_{attr}")
        expected = native_method()
        x_ptr = x.send(self.bob)
        method = getattr(x_ptr, attr)
        res_ptr = method()
        res = res_ptr.get()
        assert (res == expected).all()

    @pytest.mark.parametrize("attr", ["add", "mul"])
    def test_get_pointer_binary_method(self, attr):
        self.setUp()
        x = torch.Tensor([1, 2, 3])
        native_method = getattr(x, f"native_{attr}")
        expected = native_method(x)
        x_ptr = x.send(self.bob)
        method = getattr(x_ptr, attr)
        res_ptr = method(x_ptr)
        res = res_ptr.get()
        assert (res == expected).all()

    @pytest.mark.parametrize("attr", ["abs"])
    def test_get_pointer_to_pointer_unary_method(self, attr):
        self.setUp()
        x = torch.Tensor([1, 2, 3])
        native_method = getattr(x, f"native_{attr}")
        expected = native_method()
        x_ptr = x.send(self.bob).send(self.alice)
        method = getattr(x_ptr, attr)
        res_ptr = method()
        res = res_ptr.get().get()
        assert (res == expected).all()

    @pytest.mark.parametrize("attr", ["add", "mul"])
    def test_get_pointer_to_pointer_binary_method(self, attr):
        self.setUp()
        x = torch.Tensor([1, 2, 3])
        native_method = getattr(x, f"native_{attr}")
        expected = native_method(x)
        x_ptr = x.send(self.bob).send(self.alice)
        method = getattr(x_ptr, attr)
        res_ptr = method(x_ptr)
        res = res_ptr.get().get()
        assert (res == expected).all()

    # @pytest.mark.parametrize("attr", ["relu", "celu", "elu"])
    # def test_hook_module_functional(self, attr):
    #     self.setUp()
    #     attr = getattr(F, attr)
    #     x = torch.Tensor([1, -1, 3, 4])
    #     expected = attr(x)
    #     x_ptr = x.send(self.bob)
    #     res_ptr = attr(x_ptr)
    #     res = res_ptr.get()
    #     assert (res == expected).all()

    def test_properties(self):
        self.setUp()
        x = torch.Tensor([1, -1, 3, 4])
        assert x.is_wrapper is False
