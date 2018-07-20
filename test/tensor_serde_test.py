from unittest import TestCase
import syft as sy
import torch
import random

#hook = sy.TorchHook()

#me = hook.local_worker
#me.is_client_worker = False

#bob = sy.VirtualWorker(id="bob",hook=hook, is_client_worker=False)
#alice = sy.VirtualWorker(id="alice",hook=hook, is_client_worker=False)

#bob.add_workers([me, alice])
#alice.add_workers([me, bob])

#torch.manual_seed(1)
#random.seed(1)

class TestTensorPointerSerde(): #TestCase

    def test_tensor2unregsitered_pointer2tensor(self):
        # Tensor: Local -> Pointer (unregistered) -> Local

        x = sy.FloatTensor([1, 2, 3, 4])

        # make sure it got properly registered
        assert x.id in me._objects

        x_ptr = x.create_pointer(register=False)

        # ensure that it was NOT registered
        assert x_ptr.id not in me._objects
        x_ptr_id = x_ptr.id
        x2 = x_ptr.get()

        # make sure this returns a pointer that has been properly
        # wrapped with a FloatTensor since .get() was called on
        # a FloatTensor
        assert isinstance(x2, sy.FloatTensor)

        # make sure pointer id didn't accidentally get registered
        assert x_ptr_id not in me._objects

        # make sure the tensor that came back has the correct id
        assert x2.id == x.id

        #  since we're returning the pointer to an object that is hosted
        # locally, it should return that oject explicitly
        x += 2
        assert (x2 == torch.FloatTensor([3, 4, 5, 6])).all()

        x_ptr += 2

        # ensure that the pointer wrapper points to the same data location
        # as x
        assert (x2 == torch.FloatTensor([5, 6, 7, 8])).all()

    def test_tensor2registered_pointer2tensor(self):
        # Tensor: Local -> Pointer (unregistered) -> Local

        x = sy.FloatTensor([1, 2, 3, 4])

        # make sure it got properly registered
        assert x.id in me._objects

        x_ptr = x.create_pointer(register=True)

        # ensure that it was NOT registered
        assert x_ptr.id in me._objects
        ptr_id = x_ptr.id
        x2 = x_ptr.get()

        # ensure that it was deregistered
        assert ptr_id not in me._objects

        # make sure this returns a pointer that has been properly
        # wrapped with a FloatTensor since .get() was called on
        # a FloatTensor
        assert isinstance(x2, sy.FloatTensor)

        # make sure the tensor that came back has the correct id
        assert x2.id == x.id

        #  since we're returning the pointer to an object that is hosted
        # locally, it should return that oject explicitly
        x += 2
        assert (x2 == torch.FloatTensor([3, 4, 5, 6])).all()

        x_ptr += 2

        # ensure that the pointer wrapper points to the same data location
        # as x
        assert (x2 == torch.FloatTensor([5, 6, 7, 8])).all()
