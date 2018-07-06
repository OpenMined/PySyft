from unittest import TestCase
from syft.core.hooks import TorchHook
from syft.core.workers import VirtualWorker

import torch
from torch.autograd import Variable as Var
import torch.optim as optim
import torch.nn as nn

import json


class TestTorchTensor(TestCase):
    def test___repr__(self):

        hook = TorchHook(verbose=False)

        # stopping pep8 warning
        s = str(hook)
        s += ""

        x = torch.FloatTensor([1, 2, 3, 4, 5])
        assert x.__repr__() == '\n 1\n 2\n 3\n 4\n 5\n[torch.FloatTensor of size 5]\n'

    def test_send_tensor(self):

        hook = TorchHook(verbose=False)
        remote = VirtualWorker(id=1, hook=hook)

        x = torch.FloatTensor([1, 2, 3, 4, 5])
        x = x.send_(remote)
        assert x.id in remote._objects

    def test_get_tensor(self):

        hook = TorchHook(verbose=False)
        remote = VirtualWorker(id=1, hook=hook)

        x = torch.FloatTensor([1, 2, 3, 4, 5])
        x = x.send_(remote)

        # at this point, the remote worker should have x in its objects dict
        assert x.id in remote._objects

        assert((x.get_() == torch.FloatTensor([1, 2, 3, 4, 5])).float().mean() == 1)

        # because .get_() was called, x should no longer be in the remote worker's objects dict
        assert x.id not in remote._objects

    def test_deser_tensor(self):

        unregistered_tensor = torch.FloatTensor.deser(torch.FloatTensor, {"data": [1, 2, 3, 4, 5]})
        assert (unregistered_tensor == torch.FloatTensor([1, 2, 3, 4, 5])).float().sum() == 5

    def test_deser_tensor_from_message(self):

        hook = TorchHook(verbose=False)

        message_obj = json.loads(' {"torch_type": "torch.FloatTensor", "data": [1.0, 2.0, \
                                 3.0, 4.0, 5.0], "id": 9756847736, "owners": [1], "is_poin\
                                 ter": false}')
        obj_type = hook.guard.types_guard(message_obj['torch_type'])
        unregistered_tensor = torch.FloatTensor.deser(obj_type, message_obj)

        assert (unregistered_tensor == torch.FloatTensor([1, 2, 3, 4, 5])).float().sum() == 5

        # has not been registered
        assert unregistered_tensor.id != 9756847736

    def test_fixed_prec_ops(self):
        hook = TorchHook(verbose=False)

        x = torch.FloatTensor([1, 2, 3, 4, 5]).set_precision(7)
        y = torch.FloatTensor([1, 2, 3, 4, 5]).set_precision(3)

        assert ((x + y).free_precision() == torch.FloatTensor([2, 4, 6, 8, 10])).float().sum() == 5
        assert ((x / y).free_precision() == torch.FloatTensor([1, 1, 1, 1, 1])).float().sum() == 5
        assert ((x * y).free_precision() == torch.FloatTensor([1, 4, 9, 16, 25])).float().sum() == 5
        assert ((x - y).free_precision() == torch.FloatTensor([0, 0, 0, 0, 0])).float().sum() == 5

        x = torch.FloatTensor([1, 2, 3, 4, 5]).set_precision(3)
        y = torch.FloatTensor([1, 2, 3, 4, 5]).set_precision(7)

        assert ((x + y).free_precision() == torch.FloatTensor([2, 4, 6, 8, 10])).float().sum() == 5
        assert ((x / y).free_precision() == torch.FloatTensor([1, 1, 1, 1, 1])).float().sum() == 5
        assert ((x * y).free_precision() == torch.FloatTensor([1, 4, 9, 16, 25])).float().sum() == 5
        assert ((x - y).free_precision() == torch.FloatTensor([0, 0, 0, 0, 0])).float().sum() == 5

        x = torch.FloatTensor([1, 2, 3, 4, 5]).set_precision(3)
        y = torch.FloatTensor([1, 2, 3, 4, 5]).set_precision(3)

        assert ((x + y).free_precision() == torch.FloatTensor([2, 4, 6, 8, 10])).float().sum() == 5
        assert ((x / y).free_precision() == torch.FloatTensor([1, 1, 1, 1, 1])).float().sum() == 5
        assert ((x * y).free_precision() == torch.FloatTensor([1, 4, 9, 16, 25])).float().sum() == 5
        assert ((x - y).free_precision() == torch.FloatTensor([0, 0, 0, 0, 0])).float().sum() == 5

    def test_local_tensor_unary_methods(self):
        ''' Unit tests for methods mentioned on issue 1385
        https://github.com/OpenMined/PySyft/issues/1385'''

        x = torch.FloatTensor([1, 2, -3, 4, 5])
        assert (x.abs() == torch.FloatTensor([1, 2, 3, 4, 5])).float().sum() == 5
        assert (x.abs_() == torch.FloatTensor([1, 2, 3, 4, 5])).float().sum() == 5
        assert (x.cos() == torch.FloatTensor(
            [0.5403023, -0.41614684, -0.9899925, -0.6536436, 0.2836622])).float().sum() == 5

        assert (x.cos_() == torch.FloatTensor(
            [0.5403023, -0.41614684, -0.9899925, -0.6536436, 0.2836622])).float().sum() == 5

        x = torch.FloatTensor([1, 2, -3, 4, 5])

        assert (x.ceil() == x).float().sum() == 5
        assert (x.ceil_() == x).float().sum() == 5
        assert (x.cpu() == x).float().sum() == 5


    def test_remote_tensor_unary_methods(self):
        ''' Unit tests for methods mentioned on issue 1385
        https://github.com/OpenMined/PySyft/issues/1385'''

        hook = TorchHook(verbose=False)
        local = hook.local_worker
        remote = VirtualWorker(hook, 0)
        local.add_worker(remote)

        x = torch.FloatTensor([1, 2, -3, 4, 5]).send(remote)
        assert (x.abs().get() == torch.FloatTensor([1, 2, 3, 4, 5])).float().sum() == 5

        x = torch.FloatTensor([1, 2, -3, 4, 5]).send(remote)
        assert (x.cos().get() == torch.FloatTensor(
            [0.5403023, -0.41614684, -0.9899925, -0.6536436, 0.2836622])).float().sum() == 5
        assert (x.cos_().get() == torch.FloatTensor(
            [0.5403023, -0.41614684, -0.9899925, -0.6536436, 0.2836622])).float().sum() == 5
        x = torch.FloatTensor([1, 2, -3, 4, 5]).send(remote)
        assert (x.ceil().get() == torch.FloatTensor([1, 2, -3, 4, 5])).float().sum() == 5

        assert (x.cpu().get() == torch.FloatTensor([1, 2, -3, 4, 5])).float().sum() == 5


class TestTorchVariable(TestCase):

    def test_remote_backprop(self):

        hook = TorchHook(verbose=False)
        local = hook.local_worker
        local.verbose = False
        remote = VirtualWorker(id=1, hook=hook, verbose=False)
        local.add_worker(remote)

        x = Var(torch.ones(2, 2), requires_grad=True).send_(remote)
        x2 = Var(torch.ones(2, 2)*2, requires_grad=True).send_(remote)

        y = x * x2

        y.sum().backward()

        # remote grads should be correct
        assert (remote._objects[x2.id].grad.data == torch.ones(2, 2)).all()
        assert (remote._objects[x.id].grad.data == torch.ones(2, 2)*2).all()

        assert (y.get().data == torch.ones(2, 2)*2).all()

        assert (x.get().data == torch.ones(2, 2)).all()
        assert (x2.get().data == torch.ones(2, 2)*2).all()

        assert (x.grad.data == torch.ones(2, 2)*2).all()
        assert (x2.grad.data == torch.ones(2, 2)).all()

    def test_variable_data_attribute_bug(self):

        # previously, newly created Variable objects would lose their OpenMined given
        # attributes on the .data python objects they contain whenever the Variable
        # object is returned from a function. This bug was fixed by storing a bbackup
        # pointer to the .data object (.data_backup) so that the python object doesn't
        # get garbage collected. This test used to error out at the last line (as
        # indcated below)

        hook = TorchHook(verbose=False)
        local = hook.local_worker
        local.verbose = False

        def relu(x):
            """Rectified linear activation"""
            return torch.clamp(x, min=0.)

        def linear(x, w):
            """Linear transformation of x by w"""
            return x.mm(w)

        x = Var(torch.FloatTensor([[1, 1], [2, 2]]), requires_grad=True)
        y = Var(torch.FloatTensor([[1, 1], [2, 2]]), requires_grad=True)

        z = linear(x, y)

        # previously we had to do the following to prevent this bug
        # leaving it here for reference in case the bug returns later.
        # print(z.data.is_pointer)

        # before the bugfix, the following line would error out.
        z = relu(z)

        assert True

    def test_var_gradient_keeps_id_during_send_(self):
        # PyTorch has a tendency to delete var.grad python objects
        # and re-initialize them (resulting in new/random ids)
        # we have fixed this bug and recorded how it was fixed
        # as well as the creation of this unit test in the following
        # video (1:50:00 - 2:00:00) ish
        # https://www.twitch.tv/videos/275838386

        # this is our hook
        hook = TorchHook(verbose=False)
        local = hook.local_worker
        local.verbose = False

        remote = VirtualWorker(id=1, hook=hook, verbose=False)
        local.add_worker(remote)

        data = Var(torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]]))
        target = Var(torch.FloatTensor([[0], [0], [1], [1]]))

        model = Var(torch.zeros(2, 1), requires_grad=True)

        # generates grad objects on model
        pred = data.mm(model)
        loss = ((pred - target)**2).sum()
        loss.backward()

        # the grad's true id
        original_data_id = model.data.id + 0
        original_grad_id = model.grad.data.id + 0

        model.send_(remote)

        assert model.data.id == original_data_id
        assert model.grad.data.id == original_grad_id

    def test_send_var_with_gradient(self):

        # previously, there was a bug involving sending variables with graidents
        # to remote tensors. This bug was documented in Issue 1350
        # https://github.com/OpenMined/PySyft/issues/1350

        # this is our hook
        hook = TorchHook(verbose=False)
        local = hook.local_worker
        local.verbose = False

        remote = VirtualWorker(id=1, hook=hook, verbose=False)
        local.add_worker(remote)

        data = Var(torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]]))
        target = Var(torch.FloatTensor([[0], [0], [1], [1]]))

        model = Var(torch.zeros(2, 1), requires_grad=True)

        # generates grad objects on model
        pred = data.mm(model)
        loss = ((pred - target)**2).sum()
        loss.backward()

        # ensure that model and all (grand)children are owned by the local worker
        assert model.owners[0] == local.id
        assert model.data.owners[0] == local.id

        # if you get a failure here saying that model.grad.owners does not exist
        # check in hooks.py - _hook_new_grad(). self.grad_backup has probably either
        # been deleted or is being run at the wrong time (see comments there)
        assert model.grad.owners[0] == local.id
        assert model.grad.data.owners[0] == local.id

        # ensure that objects are not yet pointers (haven't sent it yet)
        assert not model.is_pointer
        assert not model.data.is_pointer
        assert not model.grad.is_pointer
        assert not model.grad.data.is_pointer

        model.send_(remote)

        # ensures that object ids do not change during the sending process
        assert model.owners[0].id == remote.id
        assert model.data.owners[0].id == remote.id
        assert model.grad.owners[0].id == remote.id
        assert model.grad.data.owners[0].id == remote.id

        # ensures that all local objects are now pointers
        assert model.is_pointer
        assert model.data.is_pointer
        assert model.grad.is_pointer
        assert model.grad.data.is_pointer

        # makes sure that tensors actually get sent to remote worker
        assert model.id in remote._objects
        assert model.data.id in remote._objects
        assert model.grad.id in remote._objects
        assert model.grad.data.id in remote._objects

    def test_federated_learning(self):

        hook = TorchHook(verbose=False)
        me = hook.local_worker
        me.verbose = False

        bob = VirtualWorker(id=1, hook=hook, verbose=False)
        alice = VirtualWorker(id=2, hook=hook, verbose=False)

        me.add_worker(bob)
        me.add_worker(alice)

        # create our dataset
        data = Var(torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]]))
        target = Var(torch.FloatTensor([[0], [0], [1], [1]]))

        data_bob = data[0:2].send(bob)
        target_bob = target[0:2].send(bob)

        data_alice = data[2:].send(alice)
        target_alice = target[2:].send(alice)

        # create our model
        model = nn.Linear(2, 1)

        opt = optim.SGD(params=model.parameters(), lr=0.1)

        datasets = [(data_bob, target_bob), (data_alice, target_alice)]

        for iter in range(3):

            for data, target in datasets:
                model.send(data.owners[0])

                # update the model
                model.zero_grad()
                pred = model(data)
                loss = ((pred - target)**2).sum()
                loss.backward()
                opt.step()

                model.get_()
                if(iter == 0):
                    first_loss = loss.get().data[0]

        assert loss.get().data[0] < first_loss

    def test_torch_function_on_remote_var(self):
        hook = TorchHook(verbose=False)
        me = hook.local_worker
        remote = VirtualWorker(id=2,hook=hook)
        me.add_worker(remote)

        x = Var(torch.FloatTensor([[1, 2], [3, 4]]))
        y = Var(torch.FloatTensor([[1, 2], [1, 2]]))
        x.send(remote)
        y.send(remote)
        z = torch.matmul(x, y)
        z.get()
        assert torch.equal(z, Var(torch.FloatTensor([[3, 6], [7, 14]])))

    def test_local_var_unary_methods(self):
        ''' Unit tests for methods mentioned on issue 1385
            https://github.com/OpenMined/PySyft/issues/1385'''

        x = Var(torch.FloatTensor([1, 2, -3, 4, 5]))
        assert torch.equal(x.abs(), Var(torch.FloatTensor([1, 2, 3, 4, 5])))
        assert torch.equal(x.abs_(), Var(torch.FloatTensor([1, 2, 3, 4, 5])))
        x = Var(torch.FloatTensor([1, 2, -3, 4, 5]))
        assert torch.equal(x.cos(), Var(torch.FloatTensor(
            [0.5403023, -0.41614684, -0.9899925, -0.6536436, 0.2836622])))
        x = Var(torch.FloatTensor([1, 2, -3, 4, 5]))
        assert torch.equal(x.cos_(), Var(torch.FloatTensor(
            [0.5403023, -0.41614684, -0.9899925, -0.6536436, 0.2836622])))
        x = Var(torch.FloatTensor([1, 2, -3, 4, 5]))
        assert torch.equal(x.ceil(), x)
        assert torch.equal(x.ceil_(), x)
        assert torch.equal(x.cpu(), x)


    def test_remote_var_unary_methods(self):
        ''' Unit tests for methods mentioned on issue 1385
            https://github.com/OpenMined/PySyft/issues/1385'''
        hook = TorchHook()
        local = hook.local_worker
        remote = VirtualWorker(hook, 0)
        local.add_worker(remote)

        x = Var(torch.FloatTensor([1, 2, -3, 4, 5])).send(remote)
        assert torch.equal(x.abs().get(), Var(torch.FloatTensor([1, 2, 3, 4, 5])))
        assert torch.equal(x.abs_().get(), Var(torch.FloatTensor([1, 2, 3, 4, 5])))
        assert torch.equal(x.cos().get(), Var(torch.FloatTensor(
            [0.5403023, -0.41614684, -0.9899925, -0.6536436, 0.2836622])))
        assert torch.equal(x.cos_().get(), Var(torch.FloatTensor(
            [0.5403023, -0.41614684, -0.9899925, -0.6536436, 0.2836622])))
        x = Var(torch.FloatTensor([1, 2, -3, 4, 5])).send(remote)
        assert torch.equal(x.ceil().get(), Var(torch.FloatTensor([1, 2, -3, 4, 5])))
        assert torch.equal(x.ceil_().get(), Var(torch.FloatTensor([1, 2, -3, 4, 5])))
        assert torch.equal(x.cpu().get(), Var(torch.FloatTensor([1, 2, -3, 4, 5])))