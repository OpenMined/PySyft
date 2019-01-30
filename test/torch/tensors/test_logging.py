import random

import pytest
import torch
import torch.nn.functional as F
import syft

from syft.frameworks.torch.tensors import LoggingTensor


class TestLoggingTensor(object):
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

    def test_wrap(self):
        """
        Test the .on() wrap functionality for LoggingTensor
        """
        self.setUp()
        x_tensor = torch.Tensor([1, 2, 3])
        x = LoggingTensor().on(x_tensor)
        assert isinstance(x, torch.Tensor)
        assert isinstance(x.child, LoggingTensor)
        assert isinstance(x.child.child, torch.Tensor)

    def test_overwritten_method_on_log_chain(self):
        """
        Test method call on a chain including a log tensor
        """
        self.setUp()
        # build a long chain tensor Wrapper>LoggingTensor>TorchTensor
        x_tensor = torch.Tensor([1, 2, 3])
        x = LoggingTensor().on(x_tensor)
        y = x.add(x)
        assert (y.child.child == x_tensor.add(x_tensor)).all()

    def test_method_on_log_chain(self):
        """
        Test method call on a chain including a log tensor
        """
        self.setUp()
        # build a long chain tensor Wrapper>LoggingTensor>TorchTensor
        x_tensor = torch.Tensor([1, 2, 3])
        x = LoggingTensor().on(x_tensor)
        y = x.mul(x)
        assert (y.child.child == x_tensor.mul(x_tensor)).all()

    @pytest.mark.parametrize("attr", ["relu", "celu", "elu"])
    def test_hook_module_functional_on_log_chain(self, attr):
        """
        Test torch function call on a chain including a log tensor
        """
        self.setUp()
        attr = getattr(F, attr)
        x = torch.Tensor([1, -1, 3, 4])
        expected = attr(x)
        x_log = LoggingTensor().on(x)
        res_log = attr(x_log)
        res = res_log.child.child
        assert (res == expected).all()

    def test_function_on_log_chain(self):
        """
        Test torch function call on a chain including a log tensor
        """
        self.setUp()
        x = LoggingTensor().on(torch.Tensor([1, -1, 3]))
        y = F.relu(x)
        assert (y.child.child == torch.Tensor([1, 0, 3])).all()

    def test_send_get_log_chain(self):
        """
        Test sending and getting back a chain including a logtensor
        """
        self.setUp()
        # build a long chain tensor Wrapper>LoggingTensor>TorchTensor
        x_tensor = torch.Tensor([1, 2, 3])
        x = LoggingTensor().on(x_tensor)
        x_ptr = x.send(self.bob)
        x_back = x_ptr.get()
        assert (x_back.child.child == x_tensor).all()

    def test_remote_method_on_log_chain(self):
        """
        Test remote method call on a chain including a log tensor
        """
        self.setUp()
        # build a long chain tensor Wrapper>LoggingTensor>TorchTensor
        x_tensor = torch.Tensor([1, 2, 3])
        x = LoggingTensor().on(x_tensor)
        x_ptr = x.send(self.bob)
        y_ptr = F.relu(x_ptr)
        y = y_ptr.get()
        assert (y.child.child == F.relu(x_tensor)).all()

    def test_remote_function_on_log_chain(self):
        """
        Test remote function call on a chain including a log tensor
        """
        self.setUp()
        # build a long chain tensor Wrapper>LoggingTensor>TorchTensor
        x_tensor = torch.Tensor([1, 2, 3])
        x = LoggingTensor().on(x_tensor)
        x_ptr = x.send(self.bob)
        y_ptr = x_ptr.add(x_ptr)
        y = y_ptr.get()
        assert (y.child.child == x_tensor.add(x_tensor)).all()
