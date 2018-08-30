# python -m  unittest -v test/torch_test.py


import unittest
from unittest import TestCase

import random
import syft as sy
from syft.core import utils
from syft.core.frameworks.torch import utils as torch_utils
from syft.core.frameworks.torch import encode
import torch
import torch.nn.functional as F
from torch.autograd import Variable as Var
import json

hook = sy.TorchHook(verbose=True)

me = hook.local_worker
me.is_client_worker = False

bob = sy.VirtualWorker(id="bob", hook=hook, is_client_worker=False)
alice = sy.VirtualWorker(id="alice", hook=hook, is_client_worker=False)
james = sy.VirtualWorker(id="james", hook=hook, is_client_worker=False)

me.add_workers([bob, alice, james])
bob.add_workers([me, alice, james])
alice.add_workers([me, bob, james])
james.add_workers([me, bob, alice])


class TestChainTensor(TestCase):

    def test_plus_is_minus_tensor_local(self):
        x = torch.FloatTensor([5, 6])
        y = torch.FloatTensor([3, 4])
        x = sy._PlusIsMinusTensor().on(x)
        y = sy._PlusIsMinusTensor().on(y)

        assert torch_utils.chain_print(x,
                                 display=False) == 'FloatTensor > _PlusIsMinusTensor > _LocalTensor'

        z = x.add(y)

        assert torch_utils.chain_print(z,
                                 display=False) == 'FloatTensor > _PlusIsMinusTensor > _LocalTensor'

        # cut chain for the equality check
        z.child = z.child.child
        assert torch.equal(z, torch.FloatTensor([2, 2]))

        z = torch.add(x, y)

        # cut chain for the equality check
        z.child = z.child.child
        assert torch.equal(z, torch.FloatTensor([2, 2]))

    def test_plus_is_minus_tensor_remote(self):
        x = torch.FloatTensor([5, 6])
        y = torch.FloatTensor([3, 4])
        x = sy._PlusIsMinusTensor().on(x)
        y = sy._PlusIsMinusTensor().on(y)

        id1 = random.randint(0, 10e10)
        id2 = random.randint(0, 10e10)
        x.send(bob, ptr_id=id1)
        y.send(bob, ptr_id=id2)

        z = x.add(y)
        assert torch_utils.chain_print(z, display=False) == 'FloatTensor > _PointerTensor'

        # Check chain on remote
        ptr_id = z.child.id_at_location
        assert torch_utils.chain_print(bob._objects[ptr_id].parent,
                                 display=False) == 'FloatTensor > _PlusIsMinusTensor > _LocalTensor'

        z.get()
        assert torch_utils.chain_print(z,
                                 display=False) == 'FloatTensor > _PlusIsMinusTensor > _LocalTensor'

        # cut chain for the equality check
        z.child = z.child.child
        assert torch.equal(z, torch.FloatTensor([2, 2]))

    def test_plus_is_minus_variable_local(self):
        x = sy.Variable(torch.FloatTensor([5, 6]))
        y = sy.Variable(torch.FloatTensor([3, 4]))
        x = sy._PlusIsMinusTensor().on(x)
        y = sy._PlusIsMinusTensor().on(y)

        display = 'Variable > _PlusIsMinusTensor > _LocalTensor\n' \
                  ' - FloatTensor > _PlusIsMinusTensor > _LocalTensor\n' \
                  ' - - Variable > _PlusIsMinusTensor > _LocalTensor\n' \
                  '   - FloatTensor > _PlusIsMinusTensor > _LocalTensor'

        assert torch_utils.chain_print(x, display=False) == display

        z = x.add(y)

        assert torch_utils.chain_print(z,
                                 display=False) == 'Variable > _PlusIsMinusTensor > ' \
                                                   '_LocalTensor\n - FloatTensor >' \
                                                   ' _PlusIsMinusTensor > _LocalTensor'

        # cut chain for the equality check
        z.data.child = z.data.child.child
        assert torch.equal(z.data, torch.FloatTensor([2, 2]))

        z = torch.add(x, y)

        # cut chain for the equality check
        z.data.child = z.data.child.child
        assert torch.equal(z.data, torch.FloatTensor([2, 2]))

    def test_plus_is_minus_variable_remote(self):
        x = sy.Variable(torch.FloatTensor([5, 6]))
        y = sy.Variable(torch.FloatTensor([3, 4]))
        x = sy._PlusIsMinusTensor().on(x)
        y = sy._PlusIsMinusTensor().on(y)

        id1 = random.randint(0, 10e10)
        id2 = random.randint(0, 10e10)
        id11 = random.randint(0, 10e10)
        id21 = random.randint(0, 10e10)
        x.send(bob, new_id=id1, new_data_id=id11)
        y.send(bob, new_id=id2, new_data_id=id21)

        z = x.add(y)
        assert torch_utils.chain_print(z, display=False) == 'Variable > _PointerTensor\n' \
                                                      ' - FloatTensor > _PointerTensor\n' \
                                                      ' - - Variable > _PointerTensor\n' \
                                                      '   - FloatTensor > _PointerTensor'

        assert bob._objects[z.id_at_location].owner.id == 'bob'
        assert bob._objects[z.data.id_at_location].owner.id == 'bob'

        # Check chain on remote
        ptr_id = x.child.id_at_location
        display = 'Variable > _PlusIsMinusTensor > _LocalTensor\n' \
                  ' - FloatTensor > _PlusIsMinusTensor > _LocalTensor\n' \
                  ' - - Variable > _PlusIsMinusTensor > _LocalTensor\n' \
                  '   - FloatTensor > _PlusIsMinusTensor > _LocalTensor'
        assert torch_utils.chain_print(bob._objects[ptr_id].parent, display=False) == display

        # Check chain on remote
        # TODO For now we don't reconstruct the grad chain one non-leaf variable (in our case a leaf
        # variable is a variable that we sent), because we don't care about their gradient. But if we do,
        # then this is a TODO!
        ptr_id = z.child.id_at_location
        display = 'Variable > _PlusIsMinusTensor > _LocalTensor\n' \
                  ' - FloatTensor > _PlusIsMinusTensor > _LocalTensor\n' \
                  ' - - Variable > _LocalTensor\n' \
                  '   - FloatTensor > _LocalTensor'
        assert torch_utils.chain_print(bob._objects[ptr_id].parent, display=False) == display

        z.get()
        display = 'Variable > _PlusIsMinusTensor > _LocalTensor\n' \
                  ' - FloatTensor > _PlusIsMinusTensor > _LocalTensor\n' \
                  ' - - Variable > _LocalTensor\n' \
                  '   - FloatTensor > _LocalTensor'
        assert torch_utils.chain_print(z, display=False) == display

        # cut chain for the equality check
        z.data.child = z.data.child.child
        assert torch.equal(z.data, torch.FloatTensor([2, 2]))

    def test_plus_is_minus_backward_local(self):
        x = sy.Variable(torch.FloatTensor([5, 6]), requires_grad=True)
        y = sy.Variable(torch.FloatTensor([3, 4]), requires_grad=True)
        x = sy._PlusIsMinusTensor().on(x)
        y = sy._PlusIsMinusTensor().on(y)
        z = x.add(y).sum()
        z.backward()

        # cut chain for the equality check
        x.grad.data.child = x.grad.data.child.child
        assert torch.equal(x.grad.data, torch.FloatTensor([1, 1]))

    def test_plus_is_minus_backward_remote(self):
        x = sy.Variable(torch.FloatTensor([5, 6]), requires_grad=True)
        y = sy.Variable(torch.FloatTensor([3, 4]), requires_grad=True)
        x = sy._PlusIsMinusTensor().on(x)
        y = sy._PlusIsMinusTensor().on(y)
        x.send(bob)
        y.send(bob)

        z = x.add(y).sum()
        z.backward()

        # cut chain for the equality check
        x.get()
        x.child = x.child.child

        target = sy._PlusIsMinusTensor().on(torch.FloatTensor([1, 1]))
        assert torch.equal(x.grad.data, target)



class TestTorchTensor(TestCase):

    def test___repr__(self):
        x = torch.FloatTensor([1, 2, 3, 4, 5])
        # assert x.__repr__() == '\n 1\n 2\n 3\n 4\n 5\n[torch.FloatTensor of size 5]\n'
        assert x.__repr__() == '\n 1\n 2\n 3\n 4\n 5\n[' \
                               'syft.core.frameworks.torch.tensor.FloatTensor of size 5]\n'

    def test_send_get_tensor(self):

        x = torch.FloatTensor([1, 2, 3, 4, 5])
        x_id = x.id
        ptr_id = random.randint(0, 10e10)
        x.send(bob, ptr_id=ptr_id)
        assert x_id in me._objects

        ptr = me._objects[x_id]
        assert x.child == ptr
        assert isinstance(ptr, sy._PointerTensor)
        assert ptr.id_at_location == ptr_id
        assert ptr.location.id == bob.id

        assert ptr_id in bob._objects
        remote_x = bob._objects[ptr_id]
        assert isinstance(remote_x, sy._LocalTensor)
        assert torch.equal(remote_x.child, torch.FloatTensor([1, 2, 3, 4, 5]))

        x.get()
        # Check that it's still registered
        assert x.id in me._objects
        assert torch.equal(me._objects[x.id].child, x)

        assert ((x == torch.FloatTensor([1, 2, 3, 4, 5])).all())

        # because .get_() was called, x should no longer be in the remote worker's objects dict
        assert ptr_id not in bob._objects

    def test_multiple_pointers_to_same_target(self):
        # There are two cases:
        #   - You're sending a var on a loc:id you're already pointing at -> should abort
        #   - You're pointing at the result of an in-place remote operation like:
        #       x = sy.Variable(torch.FloatTensor([1, 2, -3, 4, 5])).send(bob)
        #       y = x.abs_() # in-place operation
        #       y.get()
        #       x.send(bob) # if x.child != y.child, x will send its old pointer
        #        to bob->trigger an error
        #     You want this to work, but don't want to create a new pointer, just
        #     reuse the old one.

        # 1.
        ptr_id = random.randint(0, 10e10)
        y = torch.FloatTensor([1, 2])
        y.send(bob, ptr_id=ptr_id)
        x = torch.FloatTensor([1, 2, 3, 4, 5])
        try:
            x.send(bob, ptr_id=ptr_id)
            assert False
        except MemoryError:
            assert True

        # 2.
        x = torch.FloatTensor([1, 2, -3, 4, 5]).send(bob)
        x_id = x.id
        y = x.abs_()  # in-place operation
        assert y.child == x.child
        assert x.id == x_id
        assert y.id == x.id
        y.get()
        x.send(bob)

    def test_chain_send_get_tensor(self):

        x = torch.FloatTensor([1, 2, 3, 4, 5])
        id1 = random.randint(0, 10e10)
        id2 = random.randint(0, 10e10)
        id3 = random.randint(0, 10e10)
        x.send(bob, ptr_id=id1)
        assert id1 in bob._objects
        x.send(alice, ptr_id=id2)
        assert id2 in alice._objects
        x.send(james, ptr_id=id3)
        assert id3 in james._objects
        x.get()
        x.get()
        x.get()
        # test the get is ok
        assert torch.equal(x, torch.FloatTensor([1, 2, 3, 4, 5]))
        # Test that the remotes are empty
        assert id1 not in bob._objects
        assert id2 not in alice._objects
        assert id3 not in james._objects

    def test_add_remote_tensor(self):
        x = sy.FloatTensor([1, 2, 3, 4])
        x.send(bob, ptr_id=1000)
        x.send(alice, ptr_id=2000)
        y = sy.FloatTensor([2, 3, 4, 5])
        y.send(bob, ptr_id=1001)
        y.send(alice, ptr_id=2001)
        z = torch.add(x, y)
        z.get().get()
        assert torch.equal(z, torch.FloatTensor([3, 5, 7, 9]))

    #     def test_fixed_prec_ops(self):
    #         hook = TorchHook(verbose=False)

    #         x = torch.FloatTensor([1, 2, 3, 4, 5]).set_precision(7)
    #         y = torch.FloatTensor([1, 2, 3, 4, 5]).set_precision(3)

    #         assert ((x + y).free_precision() == torch.FloatTensor([2, 4, 6, 8, 10])).all()
    #         assert ((x / y).free_precision() == torch.FloatTensor([1, 1, 1, 1, 1])).all()
    #         assert ((x * y).free_precision() == torch.FloatTensor([1, 4, 9, 16, 25])).all()
    #         assert ((x - y).free_precision() == torch.FloatTensor([0, 0, 0, 0, 0])).all()

    #         x = torch.FloatTensor([1, 2, 3, 4, 5]).set_precision(3)
    #         y = torch.FloatTensor([1, 2, 3, 4, 5]).set_precision(7)

    #         assert ((x + y).free_precision() == torch.FloatTensor([2, 4, 6, 8, 10])).all()
    #         assert ((x / y).free_precision() == torch.FloatTensor([1, 1, 1, 1, 1])).all()
    #         assert ((x * y).free_precision() == torch.FloatTensor([1, 4, 9, 16, 25])).all()
    #         assert ((x - y).free_precision() == torch.FloatTensor([0, 0, 0, 0, 0])).all()

    #         x = torch.FloatTensor([1, 2, 3, 4, 5]).set_precision(3)
    #         y = torch.FloatTensor([1, 2, 3, 4, 5]).set_precision(3)

    #         assert ((x + y).free_precision() == torch.FloatTensor([2, 4, 6, 8, 10])).all()
    #         assert ((x / y).free_precision() == torch.FloatTensor([1, 1, 1, 1, 1])).all()
    #         assert ((x * y).free_precision() == torch.FloatTensor([1, 4, 9, 16, 25])).all()
    #         assert ((x - y).free_precision() == torch.FloatTensor([0, 0, 0, 0, 0])).all()

    def test_local_tensor_unary_methods(self):
        ''' Unit tests for methods mentioned on issue 1385
        https://github.com/OpenMined/PySyft/issues/1385'''

        x = torch.FloatTensor([1, 2, -3, 4, 5])
        assert (x.abs() == torch.FloatTensor([1, 2, 3, 4, 5])).all()
        assert (x.abs_() == torch.FloatTensor([1, 2, 3, 4, 5])).all()
        x = x.cos()
        assert (x.int() == torch.IntTensor(
            [0, 0, 0, 0, 0])).all()

        x = x.cos_()
        assert (x.int() == torch.IntTensor(
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
        assert torch.equal(torch.FloatTensor([z]), torch.FloatTensor([
            14])), "There is an issue with some Macs getting 0.0 instead, " \
                   "see https://github.com/pytorch/pytorch/issues/5609"

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

        x = torch.FloatTensor([1, 2, -3, 4, 5]).send(bob)
        assert (x.abs().get() == torch.FloatTensor([1, 2, 3, 4, 5])).all()

        x = torch.FloatTensor([1, 2, -3, 4, 5]).send(bob)
        assert (x.cos().int().get() == torch.IntTensor(
            [0, 0, 0, 0, 0])).all()
        y = x.cos_()
        assert (y.cos_().int().get() == torch.IntTensor(
            [0, 0, 0, 0, 0])).all()
        x = torch.FloatTensor([1, 2, -3, 4, 5]).send(bob)
        assert (x.ceil().get() == torch.FloatTensor([1, 2, -3, 4, 5])).all()

        assert (x.cpu().get() == torch.FloatTensor([1, 2, -3, 4, 5])).all()

    def test_remote_tensor_binary_methods(self):

        x = torch.FloatTensor([1, 2, 3, 4, 5]).send(bob)
        y = torch.FloatTensor([1, 2, 3, 4, 5]).send(bob)
        assert (torch.add(x, y).get() == torch.FloatTensor([2, 4, 6, 8, 10])).all()

        x = torch.FloatTensor([1, 2, 3, 4]).send(bob)
        y = torch.FloatTensor([[1], [2], [3], [4]]).send(bob)
        z = torch.matmul(x, y)
        assert (torch.equal(z.get(), torch.FloatTensor([30])))

        x = torch.FloatTensor([[1, 2, 3], [3, 4, 5], [5, 6, 7]]).send(bob)
        y = torch.FloatTensor([[1, 2, 3], [3, 4, 5], [5, 6, 7]]).send(bob)
        z = torch.cross(x, y, dim=1)
        assert (torch.equal(z.get(), torch.FloatTensor([[0, 0, 0], [0, 0, 0], [0, 0, 0]])))

        x = torch.FloatTensor([[1, 2, 3], [3, 4, 5], [5, 6, 7]]).send(bob)
        y = torch.FloatTensor([[1, 2, 3], [3, 4, 5], [5, 6, 7]]).send(bob)
        z = torch.dist(x, y)
        z.get()
        assert (torch.equal(z, torch.FloatTensor([0.])))

        x = torch.FloatTensor([1, 2, 3]).send(bob).send(alice)
        y = torch.FloatTensor([1, 2, 3]).send(bob).send(alice)
        z = torch.dot(x, y)
        z.get().get()
        assert torch.equal(z, torch.FloatTensor([14]))

        z = torch.eq(x, y)
        assert (torch.equal(z.get().get(), torch.ByteTensor([1, 1, 1])))

        z = torch.ge(x, y)
        assert (torch.equal(z.get().get(), torch.ByteTensor([1, 1, 1])))

    def test_local_tensor_tertiary_methods(self):

        x = torch.FloatTensor([1, 2, 3])
        y = torch.FloatTensor([1, 2, 3])
        z = torch.FloatTensor([1, 2, 3])
        assert (torch.equal(torch.addcmul(z, 2, x, y), torch.FloatTensor([3., 10., 21.])))

        x = torch.FloatTensor([1, 2, 3])
        y = torch.FloatTensor([1, 2, 3])
        z = torch.FloatTensor([1, 2, 3])
        z.addcmul_(2, x, y)
        assert (torch.equal(z, torch.FloatTensor([3., 10., 21.])))

        x = torch.FloatTensor([[1, 2]])
        y = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
        z = torch.FloatTensor([1, 2, 3])
        assert (torch.equal(torch.addmm(z, x, y), torch.FloatTensor([[10., 14., 18.]])))

    def test_remote_tensor_tertiary_methods(self):

        x = torch.FloatTensor([1, 2, 3]).send(bob)
        y = torch.FloatTensor([1, 2, 3]).send(bob)
        z = torch.FloatTensor([1, 2, 3]).send(bob)
        assert (torch.equal(torch.addcmul(z, 2, x, y).get(), torch.FloatTensor([3., 10., 21.])))

        # Uses a method
        x = torch.FloatTensor([1, 2, 3]).send(bob)
        y = torch.FloatTensor([1, 2, 3]).send(bob)
        z = torch.FloatTensor([1, 2, 3]).send(bob)
        z.addcmul_(2, x, y)
        assert (torch.equal(z.get(), torch.FloatTensor([3., 10., 21.])))

        x = torch.FloatTensor([[1, 2]]).send(bob)
        y = torch.FloatTensor([[1, 2, 3], [4, 5, 6]]).send(bob)
        z = torch.FloatTensor([1, 2, 3]).send(bob)
        assert (torch.equal(torch.addmm(z, x, y).get(), torch.FloatTensor([[10., 14., 18.]])))

    def test_local_tensor_iterable_methods(self):

        x = torch.FloatTensor([1, 2, 3])
        y = torch.FloatTensor([2, 3, 4])
        z = torch.FloatTensor([5, 6, 7])
        assert (torch.equal(torch.stack([x, y, z]),
                            torch.FloatTensor([[1, 2, 3], [2, 3, 4], [5, 6, 7]])))

        x = torch.FloatTensor([1, 2, 3])
        y = torch.FloatTensor([2, 3, 4])
        z = torch.FloatTensor([5, 6, 7])
        assert (torch.equal(torch.cat([x, y, z]), torch.FloatTensor([1, 2, 3, 2, 3, 4, 5, 6, 7])))

    def test_remote_tensor_iterable_methods(self):

        x = torch.FloatTensor([1, 2, 3]).send(bob)
        y = torch.FloatTensor([2, 3, 4]).send(bob)
        z = torch.FloatTensor([5, 6, 7]).send(bob)
        x.get()
        y.get()
        z.get()
        assert (torch.equal(torch.stack([x, y, z]),
                            torch.FloatTensor([[1, 2, 3], [2, 3, 4], [5, 6, 7]])))

        x = torch.FloatTensor([1, 2, 3]).send(bob)
        y = torch.FloatTensor([2, 3, 4]).send(bob)
        z = torch.FloatTensor([5, 6, 7]).send(bob)
        x.get()
        y.get()
        z.get()
        assert (torch.equal(torch.cat([x, y, z]), torch.FloatTensor([1, 2, 3, 2, 3, 4, 5, 6, 7])))


class TestTorchVariable(TestCase):

    def test_remote_backprop(self):

        x = sy.Variable(torch.ones(2, 2), requires_grad=True).send(bob)
        x2 = sy.Variable(torch.ones(2, 2) * 2, requires_grad=True).send(bob)

        y = x * x2

        y.sum().backward()

        # remote grads should be correct
        assert (bob._objects[x2.child.id_at_location].child.grad.data == torch.ones(2, 2)).all()
        # In particular, you can call .grad on a syft tensor, which make .child and .grad commutative
        assert (bob._objects[x2.child.id_at_location].grad.child.data == torch.ones(2, 2)).all()
        assert (bob._objects[x.child.id_at_location].child.grad.data == torch.ones(2, 2) * 2).all()

        assert (y.get().data == torch.ones(2, 2) * 2).all()

        assert (x.get().data == torch.ones(2, 2)).all()
        assert (x2.get().data == torch.ones(2, 2) * 2).all()

        assert (x.grad.data == torch.ones(2, 2) * 2).all()
        assert (x2.grad.data == torch.ones(2, 2)).all()

    def test_variable_data_attribute_bug(self):

        # previously, newly created Variable objects would lose their OpenMined given
        # attributes on the .data python objects they contain whenever the Variable
        # object is returned from a function. This bug was fixed by storing a bbackup
        # pointer to the .data object (.data_backup) so that the python object doesn't
        # get garbage collected. This test used to error out at the last line (as
        # indcated below)

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

        x = Var(torch.FloatTensor([[1, -1], [0, 1]]))
        x.send(bob)
        obj = [None, ({'marcel': (1, [1.3], x), 'proust': slice(0, 2, None)}, 3)]
        enc, t = encode.encode(obj)
        enc = json.dumps(enc)
        dec1 = encode.decode(enc, me)
        enc, t = encode.encode(dec1)
        enc = json.dumps(enc)
        dec2 = encode.decode(enc, me)
        assert dec1 == dec2

    def test_var_gradient_keeps_id_during_send_(self):
        # PyTorch has a tendency to delete var.grad python objects
        # and re-initialize them (resulting in new/random ids)
        # we have fixed this bug and recorded how it was fixed
        # as well as the creation of this unit test in the following
        # video (1:50:00 - 2:00:00) ish
        # https://www.twitch.tv/videos/275838386

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

        model.send(bob)

        assert model.data.id == original_data_id
        assert model.grad.data.id == original_grad_id

    def test_operation_with_variable_and_parameter(self):
        x = sy.Parameter(sy.FloatTensor([1]))
        y = sy.Variable(sy.FloatTensor([1]))
        z = x * y
        assert torch.equal(z, sy.Variable(sy.FloatTensor([1])))

    def test_send_var_with_gradient(self):

        # For now, we assume that var.grad.data does not get allocated
        # a pointer because it would not get used.

        data = Var(torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]]))
        target = Var(torch.FloatTensor([[0], [0], [1], [1]]))

        model = Var(torch.zeros(2, 1), requires_grad=True)

        # generates grad objects on model
        pred = data.mm(model)
        loss = ((pred - target) ** 2).sum()
        loss.backward()

        # ensure that model and all (grand)children are owned by the local worker
        assert model.owner.id == me.id
        assert model.data.owner.id == me.id

        # if you get a failure here saying that model.grad.owners does not exist
        # check in hooks.py - _hook_new_grad(). self.grad_backup has probably either
        # been deleted or is being run at the wrong time (see comments there)
        assert model.grad.owner.id == me.id
        assert model.grad.data.owner.id == me.id

        # ensure that objects are not yet pointers (haven't sent it yet)
        assert not isinstance(model.child, sy._PointerTensor)
        assert not isinstance(model.data.child, sy._PointerTensor)
        assert not isinstance(model.grad.child, sy._PointerTensor)
        assert not isinstance(model.grad.data.child, sy._PointerTensor)

        model.send(bob)

        assert model.location.id == bob.id
        assert model.data.location.id == bob.id
        assert model.grad.location.id == bob.id
        assert model.grad.data.location.id == bob.id

        # ensure that objects are not yet pointers (haven't sent it yet)
        assert isinstance(model.child, sy._PointerTensor)
        assert isinstance(model.data.child, sy._PointerTensor)
        assert isinstance(model.grad.child, sy._PointerTensor)
        assert isinstance(model.grad.data.child, sy._PointerTensor)

        assert model.id_at_location in bob._objects
        assert model.data.id_at_location in bob._objects
        assert model.grad.id_at_location in bob._objects
        assert model.grad.data.id_at_location in bob._objects

    def test_remote_optim_step(self):

        torch.manual_seed(42)

        param = []

        data = Var(torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]])).send(bob)
        target = Var(torch.FloatTensor([[0], [0], [1], [1]])).send(bob)

        model = torch.nn.Linear(2, 1)
        opt = torch.optim.SGD(params=model.parameters(), lr=0.1)

        for i in model.parameters():
            param.append(i[:])

        model.send(bob)
        model.zero_grad()
        pred = model(data)
        loss = ((pred - target) ** 2).sum()
        loss.backward()
        opt.step()

        model.get()
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
        # hook = TorchHook(verbose=False)
        # me = hook.local_worker
        # me.verbose = False
        #
        # bob = VirtualWorker(id=1, hook=hook, verbose=False)
        # alice = VirtualWorker(id=2, hook=hook, verbose=False)

        # me.add_worker(bob)
        # me.add_worker(alice)

        # create our dataset
        data = Var(torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]]))
        target = Var(torch.FloatTensor([[0], [0], [1], [1]]))

        data_bob = (data[0:2] + 0).send(bob)
        target_bob = (target[0:2] + 0).send(bob)

        data_alice = data[2:].send(alice)
        target_alice = target[2:].send(alice)

        # create our model
        model = torch.nn.Linear(2, 1)

        opt = torch.optim.SGD(params=model.parameters(), lr=0.1)

        datasets = [(data_bob, target_bob), (data_alice, target_alice)]

        for iter in range(2):

            for data, target in datasets:
                model.send(data.location)

                # update the model
                model.zero_grad()
                pred = model(data)
                loss = ((pred - target)**2).sum()
                loss.backward()
                opt.step()

                model.get()
                if(iter == 1):
                    final_loss = loss.get().data[0]

        assert (final_loss - 0.18085284531116486) < 0.001

    def test_torch_function_on_remote_var(self):
        x = sy.Variable(torch.FloatTensor([[1, 2], [3, 4]]))
        y = sy.Variable(torch.FloatTensor([[1, 2], [1, 2]]))
        x.send(bob)
        y.send(bob)
        z = torch.matmul(x, y)
        z.get()
        assert torch.equal(z, sy.Variable(torch.FloatTensor([[3, 6], [7, 14]])))

    def test_torch_function_with_multiple_input_on_remote_var(self):
        x = sy.Variable(torch.FloatTensor([1, 2]))
        y = sy.Variable(torch.FloatTensor([3, 4]))
        x.send(bob)
        y.send(bob)
        z = torch.stack([x, y])
        z.get()
        assert torch.equal(z, sy.Variable(torch.FloatTensor([[1, 2], [3, 4]])))

    def test_torch_function_with_multiple_output_on_remote_var(self):
        x = sy.Variable(torch.FloatTensor([[1, 2], [4, 3], [5, 6]]))
        x.send(bob)
        y, z = torch.max(x, 1)
        y.get()
        assert torch.equal(y, sy.Variable(torch.FloatTensor([2, 4, 6])))

    def test_torch_F_relu_on_remote_var(self):
        x = sy.Variable(torch.FloatTensor([[1, -1], [-1, 1]]))
        x.send(bob)
        x = F.relu(x)
        x.get()
        assert torch.equal(x, sy.Variable(torch.FloatTensor([[1, 0], [0, 1]])))

    def test_torch_F_conv2d_on_remote_var(self):
        x = sy.Variable(torch.FloatTensor([[[[1, -1, 2], [-1, 0, 1], [1, 0, -2]]]]))
        x.send(bob)
        weight = torch.nn.Parameter(torch.FloatTensor([[[[1, -1], [-1, 1]]]]))
        bias = torch.nn.Parameter(torch.FloatTensor([0]))
        weight.send(bob)
        bias.send(bob)
        conv = F.conv2d(x, weight, bias, stride=(1, 1))
        conv.get()
        expected_conv = sy.Variable(torch.FloatTensor([[[[3, -2], [-2, -3]]]]))
        assert torch.equal(conv, expected_conv)

    def test_torch_nn_conv2d_on_remote_var(self):

        x = sy.Variable(torch.FloatTensor([[[[1, -1, 2], [-1, 0, 1], [1, 0, -2]]]]))
        x.send(bob)
        convolute = torch.nn.Conv2d(1, 1, 2, stride=1, padding=0)
        convolute.weight = torch.nn.Parameter(torch.FloatTensor([[[[1, -1], [-1, 1]]]]))
        convolute.bias = torch.nn.Parameter(torch.FloatTensor([0]))
        convolute.send(bob)
        conv = convolute(x)
        conv.get()
        expected_conv = sy.Variable(torch.FloatTensor([[[[3, -2], [-2, -3]]]]))
        assert torch.equal(conv, expected_conv)

    def test_local_var_unary_methods(self):
        ''' Unit tests for methods mentioned on issue 1385
            https://github.com/OpenMined/PySyft/issues/1385'''

        x = sy.Variable(torch.FloatTensor([1, 2, -3, 4, 5]))
        assert torch.equal(x.abs(), sy.Variable(torch.FloatTensor([1, 2, 3, 4, 5])))
        assert torch.equal(x.abs_(), sy.Variable(torch.FloatTensor([1, 2, 3, 4, 5])))
        x = sy.Variable(torch.FloatTensor([1, 2, -3, 4, 5]))
        assert torch.equal(x.cos().int(), sy.Variable(torch.IntTensor(
            [0, 0, 0, 0, 0])))
        x = sy.Variable(torch.FloatTensor([1, 2, -3, 4, 5]))
        assert torch.equal(x.cos_().int(), sy.Variable(torch.IntTensor(
            [0, 0, 0, 0, 0])))
        x = sy.Variable(torch.FloatTensor([1, 2, -3, 4, 5]))
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
        x = sy.Variable(torch.FloatTensor([1, 2, 3, 4, 5]))
        y = sy.Variable(torch.FloatTensor([1, 2, 3, 4, 5]))
        assert torch.equal(x.add_(y), sy.Variable(torch.FloatTensor([2, 4, 6, 8, 10])))
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

        x = sy.Variable(torch.FloatTensor([1, 2, -3, 4, 5])).send(bob)
        assert torch.equal(x.abs().get(), sy.Variable(torch.FloatTensor([1, 2, 3, 4, 5])))
        assert torch.equal(x.abs_().get(), sy.Variable(torch.FloatTensor([1, 2, 3, 4, 5])))
        x = sy.Variable(torch.FloatTensor([1, 2, -3, 4, 5])).send(bob)
        assert torch.equal(x.cos().int().get(), sy.Variable(torch.IntTensor(
            [0, 0, 0, 0, 0])))
        assert torch.equal(x.cos_().int().get(), sy.Variable(torch.IntTensor(
            [0, 0, 0, 0, 0])))
        x = sy.Variable(torch.FloatTensor([1, 2, -3, 4, 5])).send(bob)
        assert torch.equal(x.ceil().get(), sy.Variable(torch.FloatTensor([1, 2, -3, 4, 5])))
        assert torch.equal(x.ceil_().get(), sy.Variable(torch.FloatTensor([1, 2, -3, 4, 5])))
        x = sy.Variable(torch.FloatTensor([1, 2, -3, 4, 5])).send(bob)
        assert torch.equal(x.cpu().get(), sy.Variable(torch.FloatTensor([1, 2, -3, 4, 5])))

    def test_remote_var_binary_methods(self):
        ''' Unit tests for methods mentioned on issue 1385
            https://github.com/OpenMined/PySyft/issues/1385'''

        x = sy.Variable(torch.FloatTensor([1, 2, 3, 4])).send(bob)
        y = sy.Variable(torch.FloatTensor([[1, 2, 3, 4]])).send(bob)
        z = torch.matmul(x, y.t())
        assert (torch.equal(z.get(), sy.Variable(torch.FloatTensor([30]))))
        z = torch.add(x, y)
        assert (torch.equal(z.get(), sy.Variable(torch.FloatTensor([[2, 4, 6, 8]]))))
        x = sy.Variable(torch.FloatTensor([1, 2, 3, 4, 5])).send(bob)
        y = sy.Variable(torch.FloatTensor([1, 2, 3, 4, 5])).send(bob)
        assert torch.equal(x.add_(y).get(), sy.Variable(torch.FloatTensor([2, 4, 6, 8, 10])))
        x = sy.Variable(torch.FloatTensor([[1, 2, 3], [3, 4, 5], [5, 6, 7]])).send(bob)
        y = sy.Variable(torch.FloatTensor([[1, 2, 3], [3, 4, 5], [5, 6, 7]])).send(bob)
        z = torch.cross(x, y, dim=1)
        assert (
            torch.equal(z.get(), sy.Variable(torch.FloatTensor([[0, 0, 0], [0, 0, 0], [0, 0, 0]]))))
        x = sy.Variable(torch.FloatTensor([[1, 2, 3], [3, 4, 5], [5, 6, 7]])).send(bob)
        y = sy.Variable(torch.FloatTensor([[1, 2, 3], [3, 4, 5], [5, 6, 7]])).send(bob)
        z = torch.dist(x, y)
        assert (torch.equal(z.get(), sy.Variable(torch.FloatTensor([0.]))))
        x = sy.Variable(torch.FloatTensor([1, 2, 3])).send(bob)
        y = sy.Variable(torch.FloatTensor([1, 2, 3])).send(bob)
        z = torch.dot(x, y)
        assert (torch.equal(z.get(), sy.Variable(torch.FloatTensor([14]))))
        z = torch.eq(x, y)
        assert (torch.equal(z.get(), sy.Variable(torch.ByteTensor([1, 1, 1]))))
        z = torch.ge(x, y)
        assert (torch.equal(z.get(), sy.Variable(torch.ByteTensor([1, 1, 1]))))


if __name__ == '__main__':
    unittest.main()
