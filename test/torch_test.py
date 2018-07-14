from unittest import TestCase
from syft.core.hooks import TorchHook
from syft.core.workers import VirtualWorker
from syft.core import utils

import torch
from torch.autograd import Variable as Var
import torch.nn.functional as F
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

        assert((x.get_() == torch.FloatTensor([1, 2, 3, 4, 5])).all())

        # because .get_() was called, x should no longer be in the remote worker's objects dict
        assert x.id not in remote._objects

    def test_deser_tensor(self):

        unregistered_tensor = torch.FloatTensor.deser(torch.FloatTensor, {"data": [1, 2, 3, 4, 5]})
        assert (unregistered_tensor == torch.FloatTensor([1, 2, 3, 4, 5])).all()

    def test_deser_tensor_from_message(self):

        hook = TorchHook(verbose=False)

        message_obj = json.loads(' {"torch_type": "torch.FloatTensor", "data": [1.0, 2.0, \
                                 3.0, 4.0, 5.0], "id": 9756847736, "owners": [1], "is_poin\
                                 ter": false}')
        obj_type = hook.guard.types_guard(message_obj['torch_type'])
        unregistered_tensor = torch.FloatTensor.deser(obj_type, message_obj)

        assert (unregistered_tensor == torch.FloatTensor([1, 2, 3, 4, 5])).all()

        # has not been registered
        assert unregistered_tensor.id != 9756847736

    def test_fixed_prec_ops(self):
        hook = TorchHook(verbose=False)

        x = torch.FloatTensor([1, 2, 3, 4, 5]).set_precision(7)
        y = torch.FloatTensor([1, 2, 3, 4, 5]).set_precision(3)

        assert ((x + y).free_precision() == torch.FloatTensor([2, 4, 6, 8, 10])).all()
        assert ((x / y).free_precision() == torch.FloatTensor([1, 1, 1, 1, 1])).all()
        assert ((x * y).free_precision() == torch.FloatTensor([1, 4, 9, 16, 25])).all()
        assert ((x - y).free_precision() == torch.FloatTensor([0, 0, 0, 0, 0])).all()

        x = torch.FloatTensor([1, 2, 3, 4, 5]).set_precision(3)
        y = torch.FloatTensor([1, 2, 3, 4, 5]).set_precision(7)

        assert ((x + y).free_precision() == torch.FloatTensor([2, 4, 6, 8, 10])).all()
        assert ((x / y).free_precision() == torch.FloatTensor([1, 1, 1, 1, 1])).all()
        assert ((x * y).free_precision() == torch.FloatTensor([1, 4, 9, 16, 25])).all()
        assert ((x - y).free_precision() == torch.FloatTensor([0, 0, 0, 0, 0])).all()

        x = torch.FloatTensor([1, 2, 3, 4, 5]).set_precision(3)
        y = torch.FloatTensor([1, 2, 3, 4, 5]).set_precision(3)

        assert ((x + y).free_precision() == torch.FloatTensor([2, 4, 6, 8, 10])).all()
        assert ((x / y).free_precision() == torch.FloatTensor([1, 1, 1, 1, 1])).all()
        assert ((x * y).free_precision() == torch.FloatTensor([1, 4, 9, 16, 25])).all()
        assert ((x - y).free_precision() == torch.FloatTensor([0, 0, 0, 0, 0])).all()

    def test_local_tensor_unary_methods(self):
        ''' Unit tests for methods mentioned on issue 1385
        https://github.com/OpenMined/PySyft/issues/1385'''

        x = torch.FloatTensor([1, 2, -3, 4, 5])
        assert (x.abs() == torch.FloatTensor([1, 2, 3, 4, 5])).all()
        assert (x.abs_() == torch.FloatTensor([1, 2, 3, 4, 5])).all()
        x = x.cos()
        assert (x.int().get() == torch.IntTensor(
            [0, 0, 0, 0, 0])).all()

        x = x.cos_()
        assert (x.int().get() == torch.IntTensor(
            [0, 0, 0, 0, 0])).all()

        x = torch.FloatTensor([1, 2, -3, 4, 5])

        assert (x.ceil() == x).all()
        assert (x.ceil_() == x).all()
        assert (x.cpu() == x).all()

    def test_local_tensor_binary_methods(self):
        ''' Unit tests for methods mentioned on issue 1385
        https://github.com/OpenMined/PySyft/issues/1385'''

        x = torch.FloatTensor([1, 2, 3, 4])
        y = torch.FloatTensor([[1, 2, 3, 4]])
        z = torch.matmul(x, y.t())
        assert (torch.equal(z, torch.FloatTensor([30])))

        z = torch.add(x, y)
        assert (torch.equal(z, torch.FloatTensor([[2, 4, 6, 8]])))

        x = torch.FloatTensor([[1, 2, 3], [3, 4, 5], [5, 6, 7]])
        y = torch.FloatTensor([[1, 2, 3], [3, 4, 5], [5, 6, 7]])
        z = torch.cross(x, y, dim=1)
        assert (torch.equal(z, torch.FloatTensor([[0, 0, 0], [0, 0, 0], [0, 0, 0]])))

        x = torch.FloatTensor([[1, 2, 3], [3, 4, 5], [5, 6, 7]])
        y = torch.FloatTensor([[1, 2, 3], [3, 4, 5], [5, 6, 7]])
        z = torch.dist(x, y)
        assert (torch.equal(torch.FloatTensor([z]), torch.FloatTensor([0])))

        x = torch.FloatTensor([1, 2, 3])
        y = torch.FloatTensor([1, 2, 3])
        z = torch.dot(x, y)
        # There is an issue with some Macs getting 0.0 instead
        # Solved here: https://github.com/pytorch/pytorch/issues/5609
        assert torch.equal(torch.FloatTensor([z]), torch.FloatTensor([14]))

        z = torch.eq(x, y)
        assert (torch.equal(z, torch.ByteTensor([1, 1, 1])))

        z = torch.ge(x, y)
        assert (torch.equal(z, torch.ByteTensor([1, 1, 1])))

        x = torch.FloatTensor([1, 2, 3, 4, 5])
        y = torch.FloatTensor([1, 2, 3, 4, 5])
        assert (x.add_(y) == torch.FloatTensor([2, 4, 6, 8, 10])).all()

    def test_remote_tensor_unary_methods(self):
        ''' Unit tests for methods mentioned on issue 1385
        https://github.com/OpenMined/PySyft/issues/1385'''

        hook = TorchHook(verbose=False)
        local = hook.local_worker
        remote = VirtualWorker(id=2,hook=hook)
        local.add_worker(remote)

        x = torch.FloatTensor([1, 2, -3, 4, 5]).send(remote)
        assert (x.abs().get() == torch.FloatTensor([1, 2, 3, 4, 5])).all()

        x = torch.FloatTensor([1, 2, -3, 4, 5]).send(remote)
        assert (x.cos().int().get() == torch.IntTensor(
            [0, 0, 0, 0, 0])).all()
        y = x.cos_()
        assert (y.cos_().int().get() == torch.IntTensor(
            [0, 0, 0, 0, 0])).all()
        x = torch.FloatTensor([1, 2, -3, 4, 5]).send(remote)
        assert (x.ceil().get() == torch.FloatTensor([1, 2, -3, 4, 5])).all()

        assert (x.cpu().get() == torch.FloatTensor([1, 2, -3, 4, 5])).all()


    def test_remote_tensor_binary_methods(self):

        hook = TorchHook(verbose = False)
        local = hook.local_worker
        remote = VirtualWorker(hook, 0)
        local.add_worker(remote)

        x = torch.FloatTensor([1, 2, 3, 4, 5]).send(remote)
        y = torch.FloatTensor([1, 2, 3, 4, 5]).send(remote)
        assert (x.add_(y).get() == torch.FloatTensor([2,4,6,8,10])).all()

        x = torch.FloatTensor([1, 2, 3, 4]).send(remote)
        y = torch.FloatTensor([[1, 2, 3, 4]]).send(remote)
        z = torch.matmul(x, y.t())
        assert (torch.equal(z.get(), torch.FloatTensor([30])))

        z = torch.add(x, y)
        assert (torch.equal(z.get(), torch.FloatTensor([[2, 4, 6, 8]])))

        x = torch.FloatTensor([[1, 2, 3], [3, 4, 5], [5, 6, 7]]).send(remote)
        y = torch.FloatTensor([[1, 2, 3], [3, 4, 5], [5, 6, 7]]).send(remote)
        z = torch.cross(x, y, dim=1)
        assert (torch.equal(z.get(), torch.FloatTensor([[0, 0, 0], [0, 0, 0], [0, 0, 0]])))

        x = torch.FloatTensor([[1, 2, 3], [3, 4, 5], [5, 6, 7]]).send(remote)
        y = torch.FloatTensor([[1, 2, 3], [3, 4, 5], [5, 6, 7]]).send(remote)
        z = torch.dist(x, y)
        t = torch.FloatTensor([z])
        assert (torch.equal(t, torch.FloatTensor([0.])))

        x = torch.FloatTensor([1, 2, 3]).send(remote)
        y = torch.FloatTensor([1, 2, 3]).send(remote)
        z = torch.dot(x, y)
        t = torch.FloatTensor([z])
        assert torch.equal(t, torch.FloatTensor([14]))

        z = torch.eq(x, y)
        assert (torch.equal(z.get(), torch.ByteTensor([1, 1, 1])))

        z = torch.ge(x, y)
        assert (torch.equal(z.get(), torch.ByteTensor([1, 1, 1])))


    def test_local_tensor_tertiary_methods(self):

        x = torch.FloatTensor([1, 2, 3])
        y = torch.FloatTensor([1, 2, 3])
        z = torch.FloatTensor([1, 2, 3])
        assert (torch.equal(torch.addcmul(z, 2, x, y), torch.FloatTensor([3.,  10.,  21.])))

        x = torch.FloatTensor([1, 2, 3])
        y = torch.FloatTensor([1, 2, 3])
        z = torch.FloatTensor([1, 2, 3])
        z.addcmul_(2, x, y)
        assert (torch.equal(z, torch.FloatTensor([3., 10., 21.])))

        x = torch.FloatTensor([[1, 2]])
        y = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
        z = torch.FloatTensor([1, 2, 3])
        assert(torch.equal(torch.addmm(z, x, y), torch.FloatTensor([[10., 14., 18.]])))

    def test_remote_tensor_tertiary_methods(self):

        hook = TorchHook(verbose=False)
        local = hook.local_worker
        remote = VirtualWorker(hook, 1)
        local.add_worker(remote)

        x = torch.FloatTensor([1, 2, 3]).send(remote)
        y = torch.FloatTensor([1, 2, 3]).send(remote)
        z = torch.FloatTensor([1, 2, 3]).send(remote)
        assert (torch.equal(torch.addcmul(z,  2, x, y).get(), torch.FloatTensor([3., 10., 21.])))

        x = torch.FloatTensor([1, 2, 3]).send(remote)
        y = torch.FloatTensor([1, 2, 3]).send(remote)
        z = torch.FloatTensor([1, 2, 3]).send(remote)
        z.addcmul_(2, x, y)
        assert (torch.equal(z.get(), torch.FloatTensor([3., 10., 21.])))

        x = torch.FloatTensor([[1, 2]]).send(remote)
        y = torch.FloatTensor([[1, 2, 3], [4, 5, 6]]).send(remote)
        z = torch.FloatTensor([1, 2, 3]).send(remote)
        assert (torch.equal(torch.addmm(z, x, y).get(), torch.FloatTensor([[10., 14., 18.]])))

    def test_local_tensor_iterable_methods(self):

        x = torch.FloatTensor([1, 2, 3])
        y = torch.FloatTensor([2, 3, 4])
        z = torch.FloatTensor([5, 6, 7])
        assert(torch.equal(torch.stack([x, y, z]), torch.FloatTensor([[1, 2, 3], [2, 3, 4], [5, 6, 7]])))

        x = torch.FloatTensor([1, 2, 3])
        y = torch.FloatTensor([2, 3, 4])
        z = torch.FloatTensor([5, 6, 7])
        assert (torch.equal(torch.cat([x, y, z]), torch.FloatTensor([1, 2, 3, 2, 3, 4, 5, 6, 7])))

    def test_remote_tensor_iterable_methods(self):

        hook = TorchHook(verbose=False)
        local = hook.local_worker
        remote = VirtualWorker(hook, 1)
        local.add_worker(remote)

        x = torch.FloatTensor([1, 2, 3]).send(remote)
        y = torch.FloatTensor([2, 3, 4]).send(remote)
        z = torch.FloatTensor([5, 6, 7]).send(remote)

        assert(torch.equal(torch.stack([x, y, z]).get(), torch.FloatTensor([[1, 2, 3], [2, 3, 4], [5, 6, 7]])))

        x = torch.FloatTensor([1, 2, 3]).send(remote)
        y = torch.FloatTensor([2, 3, 4]).send(remote)
        z = torch.FloatTensor([5, 6, 7]).send(remote)

        assert (torch.equal(torch.cat([x, y, z]).get(), torch.FloatTensor([1, 2, 3, 2, 3, 4, 5, 6, 7])))

    def test_local_tensor_multi_var_methods(self):
        x = torch.FloatTensor([[1, 2], [2, 3], [5, 6]])
        t, s = torch.max(x, 1)
        assert (t == torch.FloatTensor([2, 3, 6])).float().sum() == 3
        assert (s == torch.LongTensor([1, 1, 1])).float().sum() == 3

        x = torch.FloatTensor([[0, 0], [1, 1]])
        y, z = torch.eig(x, True)
        assert (y == torch.FloatTensor([[1, 0], [0, 0]])).all()
        assert (torch.equal(z == torch.FloatTensor([[0, 0], [1, 0]]), torch.ByteTensor([[1, 0], [1, 0]])))

        x = torch.FloatTensor([[0, 0], [1, 0]])
        y, z = torch.qr(x)
        assert (y == torch.FloatTensor([[0, -1], [-1, 0]])).all()
        assert (z == torch.FloatTensor([[-1, 0], [0, 0]])).all()

        x = torch.arange(1, 6)
        y, z = torch.kthvalue(x, 4)
        assert (y == torch.FloatTensor([4])).all()
        assert (z == torch.LongTensor([3])).all()

        x = torch.zeros(3, 3)
        w, y, z = torch.svd(x)
        assert (w == torch.FloatTensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])).all()
        assert (y == torch.FloatTensor([0, 0, 0])).all()
        assert (z == torch.FloatTensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])).all()

    def test_remote_tensor_multi_var_methods(self):
        hook = TorchHook(verbose=False)
        local = hook.local_worker
        remote = VirtualWorker(hook, 1)
        local.add_worker(remote)

        x = torch.FloatTensor([[1, 2], [4, 3], [5, 6]])
        x.send(remote)
        y, z = torch.max(x, 1)
        assert torch.equal(y.get(), torch.FloatTensor([2, 4, 6]))
        assert torch.equal(z.get(), torch.LongTensor([1, 0, 1]))

        x = torch.FloatTensor([[0, 0], [1, 0]]).send(remote)
        y, z = torch.qr(x)
        assert (y.get() == torch.FloatTensor([[0, -1], [-1, 0]])).all()
        assert (z.get() == torch.FloatTensor([[-1, 0], [0, 0]])).all()

        x = torch.arange(1, 6).send(remote)
        y, z = torch.kthvalue(x, 4)
        assert (y.get() == torch.FloatTensor([4])).all()
        assert (z.get() == torch.LongTensor([3])).all()

        x = torch.FloatTensor([[0, 0], [1, 1]]).send(remote)
        y, z = torch.eig(x, True)
        assert (y.get() == torch.FloatTensor([[1, 0], [0, 0]])).all()
        assert ((z.get() == torch.FloatTensor([[0, 0], [1, 0]])) == torch.ByteTensor([[1, 0], [1, 0]])).all()

        x = torch.zeros(3, 3).send(remote)
        w, y, z = torch.svd(x)
        assert (w.get() == torch.FloatTensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])).all()
        assert (y.get() == torch.FloatTensor([0, 0, 0])).all()
        assert (z.get() == torch.FloatTensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])).all()


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

    def test_encode_decode_json_python(self):
        """
            Test that the python objects are correctly encoded and decoded in
            json with our encoder/JSONDecoder.
            The main focus is on non-serializable objects, such as torch Variable
            or tuple, or even slice().
        """
        hook = TorchHook(verbose=False)
        local = hook.local_worker
        remote = VirtualWorker(id=1, hook=hook)
        local.add_worker(remote)

        encoder = utils.PythonEncoder(retrieve_tensorvar=True)
        decoder = utils.PythonJSONDecoder(remote)
        x = Var(torch.FloatTensor([[1, -1],[0,1]]))
        x.send(remote)
        # Note that there is two steps of encoding/decoding because the first
        # transforms `Variable containing:[torch.FloatTensor - Locations:[
        # <syft.core.workers.virtual.VirtualWorker id:2>]]` into
        # Variable containing:[torch.FloatTensor - Locations:[2]]`
        obj = [None, ({'marcel': (1, [1.3], x), 'proust': slice(0, 2, None)}, 3)]
        enc, t = encoder.encode(obj)
        enc = json.dumps(enc)
        dec1 = decoder.decode(enc)
        enc, t = encoder.encode(dec1)
        enc = json.dumps(enc)
        dec2 = decoder.decode(enc)
        assert dec1 == dec2

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
        assert model.owners[0].id == local.id
        assert model.data.owners[0].id == local.id

        # if you get a failure here saying that model.grad.owners does not exist
        # check in hooks.py - _hook_new_grad(). self.grad_backup has probably either
        # been deleted or is being run at the wrong time (see comments there)
        assert model.grad.owners[0].id == local.id
        assert model.grad.data.owners[0].id == local.id

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

    def test_remote_optim_step(self):

        torch.manual_seed(42)
        hook = TorchHook(verbose=False)
        local = hook.local_worker
        local.verbose = False
        remote = VirtualWorker(id=1, hook=hook, verbose=False)
        local.add_worker(remote)
        param = []

        data = Var(torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]])).send(remote)
        target = Var(torch.FloatTensor([[0], [0], [1], [1]])).send(remote)

        model = nn.Linear(2, 1)
        opt = optim.SGD(params=model.parameters(), lr=0.1)

        for i in model.parameters():
            param.append(i[:])

        model.send_(remote)
        model.zero_grad()
        pred = model(data)
        loss = ((pred - target) ** 2).sum()
        loss.backward()
        opt.step()

        model.get_()
        for i in model.parameters():
            param.append(i[:])

        x = []
        for i in param:
            if type(i.data[0]) != float:
                x.append(i.data[0][0])
                x.append(i.data[0][1])
            else:
                x.append(i.data[0])

        y = [0.5406, 0.5869, -0.16565567255020142, 0.6732, 0.5103, -0.0841369703412056]

        assert (self.assertAlmostEqual(X,Y) for X,Y in zip(x,y))

    def test_federated_learning(self):

        torch.manual_seed(42)
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

        for iter in range(2):

            for data, target in datasets:
                model.send(data.owners[0])

                # update the model
                model.zero_grad()
                pred = model(data)
                loss = ((pred - target)**2).sum()
                loss.backward()
                opt.step()

                model.get_()
                if(iter == 1):
                    final_loss = loss.get().data[0]

        assert final_loss == 0.18085284531116486

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

    def test_torch_function_with_multiple_input_on_remote_var(self):
        hook = TorchHook(verbose=False)
        me = hook.local_worker
        remote = VirtualWorker(id=2,hook=hook)
        me.add_worker(remote)

        x = Var(torch.FloatTensor([1,2]))
        y = Var(torch.FloatTensor([3,4]))
        x.send(remote)
        y.send(remote)
        z = torch.stack([x,y])
        z.get()
        assert torch.equal(z, Var(torch.FloatTensor([[1, 2], [3, 4]])))

    def test_torch_function_with_multiple_output_on_remote_var(self):
        hook = TorchHook(verbose=False)
        me = hook.local_worker
        remote = VirtualWorker(id=2,hook=hook)
        me.add_worker(remote)

        x = Var(torch.FloatTensor([[1,2],[4,3],[5,6]]))
        x.send(remote)
        y, z = torch.max(x, 1)
        y.get()
        assert torch.equal(y, Var(torch.FloatTensor([2, 4, 6])))

    def test_torch_F_relu_on_remote_var(self):
        hook = TorchHook(verbose=False)
        me = hook.local_worker
        remote = VirtualWorker(id=2,hook=hook)
        me.add_worker(remote)

        x = Var(torch.FloatTensor([[1, -1], [-1, 1]]))
        x.send(remote)
        x = F.relu(x)
        x.get()
        assert torch.equal(x, Var(torch.FloatTensor([[1, 0], [0, 1]])))

    def test_torch_F_conv2d_on_remote_var(self):
        hook = TorchHook(verbose=False)
        me = hook.local_worker
        remote = VirtualWorker(id=2,hook=hook)
        me.add_worker(remote)

        x = Var(torch.FloatTensor([[[[1, -1, 2], [-1, 0, 1], [1, 0, -2]]]]))
        x.send(remote)
        weight = torch.nn.Parameter(torch.FloatTensor([[[[1, -1], [-1, 1]]]]))
        bias = torch.nn.Parameter(torch.FloatTensor([0]))
        weight.send(remote)
        bias.send(remote)
        conv = F.conv2d(x, weight, bias, stride=(1,1))
        conv.get()
        expected_conv = Var(torch.FloatTensor([[[[3, -2], [-2, -3]]]]))
        assert torch.equal(conv, expected_conv)

    def test_torch_nn_conv2d_on_remote_var(self):
        hook = TorchHook(verbose=False)
        me = hook.local_worker
        remote = VirtualWorker(id=2,hook=hook)
        me.add_worker(remote)

        x = Var(torch.FloatTensor([[[[1, -1, 2], [-1, 0, 1], [1, 0, -2]]]]))
        x.send(remote)
        convolute = nn.Conv2d(1, 1, 2, stride=1, padding=0)
        convolute.weight = torch.nn.Parameter(torch.FloatTensor([[[[1, -1], [-1, 1]]]]))
        convolute.bias = torch.nn.Parameter(torch.FloatTensor([0]))
        convolute.send(remote)
        conv = convolute(x)
        conv.get()
        expected_conv = Var(torch.FloatTensor([[[[3, -2], [-2, -3]]]]))
        assert torch.equal(conv, expected_conv)

    def test_local_var_unary_methods(self):
        ''' Unit tests for methods mentioned on issue 1385
            https://github.com/OpenMined/PySyft/issues/1385'''

        x = Var(torch.FloatTensor([1, 2, -3, 4, 5]))
        assert torch.equal(x.abs(), Var(torch.FloatTensor([1, 2, 3, 4, 5])))
        assert torch.equal(x.abs_(), Var(torch.FloatTensor([1, 2, 3, 4, 5])))
        x = Var(torch.FloatTensor([1, 2, -3, 4, 5]))
        assert torch.equal(x.cos().int(), Var(torch.IntTensor(
            [0, 0, 0, 0, 0])))
        x = Var(torch.FloatTensor([1, 2, -3, 4, 5]))
        assert torch.equal(x.cos_().int(), Var(torch.IntTensor(
            [0, 0, 0, 0, 0])))
        x = Var(torch.FloatTensor([1, 2, -3, 4, 5]))
        assert torch.equal(x.ceil(), x)
        assert torch.equal(x.ceil_(), x)
        assert torch.equal(x.cpu(), x)


    def test_local_var_binary_methods(self):
        ''' Unit tests for methods mentioned on issue 1385
            https://github.com/OpenMined/PySyft/issues/1385'''
        x = torch.FloatTensor([1, 2, 3, 4])
        y = torch.FloatTensor([[1, 2, 3, 4]])
        z = torch.matmul(x, y.t())
        assert (torch.equal(z, torch.FloatTensor([30])))
        z = torch.add(x, y)
        assert (torch.equal(z, torch.FloatTensor([[2, 4, 6, 8]])))
        x = torch.FloatTensor([[1, 2, 3], [3, 4, 5], [5, 6, 7]])
        y = torch.FloatTensor([[1, 2, 3], [3, 4, 5], [5, 6, 7]])
        z = torch.cross(x, y, dim=1)
        assert (torch.equal(z, torch.FloatTensor([[0, 0, 0], [0, 0, 0], [0, 0, 0]])))
        x = torch.FloatTensor([[1, 2, 3], [3, 4, 5], [5, 6, 7]])
        y = torch.FloatTensor([[1, 2, 3], [3, 4, 5], [5, 6, 7]])
        z = torch.dist(x, y)
        t = torch.FloatTensor([z])
        assert (torch.equal(t, torch.FloatTensor([0.])))
        x = torch.FloatTensor([1, 2, 3])
        y = torch.FloatTensor([1, 2, 3])
        z = torch.dot(x, y)
        t = torch.FloatTensor([z])
        assert torch.equal(t, torch.FloatTensor([14]))
        z = torch.eq(x, y)
        assert (torch.equal(z, torch.ByteTensor([1, 1, 1])))
        z = torch.ge(x, y)
        assert (torch.equal(z, torch.ByteTensor([1, 1, 1])))

    def test_remote_var_unary_methods(self):
        ''' Unit tests for methods mentioned on issue 1385
            https://github.com/OpenMined/PySyft/issues/1385'''
        hook = TorchHook(verbose=False)
        local = hook.local_worker
        remote = VirtualWorker(id=2,hook=hook)
        local.add_worker(remote)

        x = Var(torch.FloatTensor([1, 2, -3, 4, 5])).send(remote)
        assert torch.equal(x.abs().get(), Var(torch.FloatTensor([1, 2, 3, 4, 5])))
        assert torch.equal(x.abs_().get(), Var(torch.FloatTensor([1, 2, 3, 4, 5])))
        assert torch.equal(x.cos().int().get(), Var(torch.IntTensor(
            [0, 0, 0, 0, 0])))
        assert torch.equal(x.cos_().int().get(), Var(torch.IntTensor(
            [0, 0, 0, 0, 0])))
        x = Var(torch.FloatTensor([1, 2, -3, 4, 5])).send(remote)
        assert torch.equal(x.ceil().get(), Var(torch.FloatTensor([1, 2, -3, 4, 5])))
        assert torch.equal(x.ceil_().get(), Var(torch.FloatTensor([1, 2, -3, 4, 5])))
        assert torch.equal(x.cpu().get(), Var(torch.FloatTensor([1, 2, -3, 4, 5])))

    def test_local_var_binary_methods(self):
        
        x = Var(torch.FloatTensor([1, 2, 3, 4, 5]))
        y = Var(torch.FloatTensor([1, 2, 3, 4, 5]))
        assert  torch.equal(x.add_(y), Var(torch.FloatTensor([2,4,6,8,10])))

    def test_remote_var_binary_methods(self):

        hook = TorchHook()
        local = hook.local_worker
        remote = VirtualWorker(hook, 0)
        local.add_worker(remote)

        x = Var(torch.FloatTensor([1, 2, 3, 4, 5])).send(remote)
        y = Var(torch.FloatTensor([1, 2, 3, 4, 5])).send(remote)
        assert torch.equal(x.add_(y).get(),  Var(torch.FloatTensor([2,4,6,8,10])))

    def test_remote_var_binary_methods(self):
        ''' Unit tests for methods mentioned on issue 1385
            https://github.com/OpenMined/PySyft/issues/1385'''
        hook = TorchHook(verbose=False)
        local = hook.local_worker
        remote = VirtualWorker(hook, 1)
        local.add_worker(remote)

        x = Var(torch.FloatTensor([1, 2, 3, 4])).send(remote)
        y = Var(torch.FloatTensor([[1, 2, 3, 4]])).send(remote)
        z = torch.matmul(x, y.t())
        assert (torch.equal(z.get(), Var(torch.FloatTensor([30]))))
        z = torch.add(x, y)
        assert (torch.equal(z.get(), Var(torch.FloatTensor([[2, 4, 6, 8]]))))
        x = Var(torch.FloatTensor([[1, 2, 3], [3, 4, 5], [5, 6, 7]])).send(remote)
        y = Var(torch.FloatTensor([[1, 2, 3], [3, 4, 5], [5, 6, 7]])).send(remote)
        z = torch.cross(x, y, dim=1)
        assert (torch.equal(z.get(), Var(torch.FloatTensor([[0, 0, 0], [0, 0, 0], [0, 0, 0]]))))
        x = Var(torch.FloatTensor([[1, 2, 3], [3, 4, 5], [5, 6, 7]])).send(remote)
        y = Var(torch.FloatTensor([[1, 2, 3], [3, 4, 5], [5, 6, 7]])).send(remote)
        z = torch.dist(x, y)
        assert (torch.equal(z.get(), Var(torch.FloatTensor([0.]))))
        x = Var(torch.FloatTensor([1, 2, 3])).send(remote)
        y = Var(torch.FloatTensor([1, 2, 3])).send(remote)
        z = torch.dot(x, y)
        print(torch.equal(z.get(), Var(torch.FloatTensor([14]))))
        z = torch.eq(x, y)
        assert (torch.equal(z.get(), Var(torch.ByteTensor([1, 1, 1]))))
        z = torch.ge(x, y)
        assert (torch.equal(z.get(), Var(torch.ByteTensor([1, 1, 1]))))
