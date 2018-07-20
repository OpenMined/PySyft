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

    def test_floattensordata2json2floattensordata(self):
        # this tests the serialization / deserialization of the data FloatTensor
        # objects (which is different from those which are merely wrappers).

        x = sy.FloatTensor([1, 2, 3, 4, 5])

        xs = {'data': [1.0, 2.0, 3.0, 4.0, 5.0],
              'torch_type': 'syft.FloatTensor',
              'type': 'syft.core.frameworks.torch.tensor.FloatTensor'}

        assert x.ser(stop_recurse_at_torch_type=True) == xs

        x2 = sy.FloatTensor.deser(xs, register=False)

        # ensure it has no id
        assert not hasattr(x2, 'id')

        # ensure values are the same as what was serialized
        assert x2.tolist() == x.tolist()

        # ensure t-string specifies that the tensor has no children
        assert x2.__str__() == 'Empty Wrapper:\n\n 1\n 2\n 3\n 4\n' + \
               ' 5\n[syft.core.frameworks.torch.tensor.FloatTensor of size 5]\n'

        # ensure tensor has no children (x2.child is automatically genreated in this case
        # we're just making sure the generated child has no child or id.
        assert not hasattr(x2.child, 'child')
        assert not hasattr(x2.child, 'id')

    def test_localtensor2json2localtensor(self):

        xs = {'child': {'data': [1.0, 2.0, 3.0, 4.0],
                        'torch_type': 'syft.FloatTensor',
                        'type': 'syft.core.frameworks.torch.tensor.FloatTensor'},
              'id': 8684158308,
              'owner': 0,
              'torch_type': 'syft.FloatTensor',
              'type': 'syft.core.frameworks.torch.tensor._LocalTensor'}

        x = sy.FloatTensor([1, 2, 3, 4])
        x.child.id = xs['id']

        # check that serialization is correct
        assert xs == x.child.ser()

        # reset ID for further testing
        xs['id'] = 54321

        x = sy._LocalTensor.deser(xs, register=False)

        # correct id
        assert x.id == xs['id']

        # correct owner
        assert x.owner.id == xs['owner']

        # correct type
        assert type(x).__name__ == xs['type'].split(".")[-1]

        # correct size
        assert len(x) == 4

        # correct data
        assert (x[0:4] == sy.FloatTensor([1, 2, 3, 4])).all()

        # object shouldn't be in registry yet because we
        # set register=False
        assert x.id not in me._objects

        # deser again except this time allow it to be registered
        x2 = sy._LocalTensor.deser(xs)

        assert x.id in me._objects

        try:
            x2 = sy._LocalTensor.deser(xs)
            assert False
        except Exception as e:
            assert "Cannot deserialize and register a tensor that already exists." in str(e)

        # repeat test again except this time don't pass in a child at all
        # let the object initialize an empty one instead

        del xs['child']
        xs['id'] = 12345

        x3 = sy._LocalTensor.deser(xs)

        assert x3.id == 12345
        assert x3.owner.id == 0
        assert x3.id in x.owner._objects
        assert type(x3).__name__ == xs['type'].split(".")[-1]
        assert len(x3) == 0

    def test_floattensor2json2floattensor(self):

        xs = {'child': {'data': [1.0, 2.0, 3.0, 4.0, 5.0],
                        'torch_type': 'syft.FloatTensor',
                        'type': 'syft.core.frameworks.torch.tensor.FloatTensor'},
              'id': 234152,
              'owner': 0,
              'torch_type': 'syft.FloatTensor',
              'type': 'syft.core.frameworks.torch.tensor._LocalTensor'}

        x = sy.FloatTensor([1, 2, 3, 4, 5])
        x.child.id = 234152

        # test that serialization happens correctly
        assert x.ser() == xs

        # initialize tensor without registering it
        x2 = sy.FloatTensor.deser(xs, register=False)

        # check id and owner are correct
        assert x2.id == 234152
        assert x2.owner.id == 0

        # make sure it works (can do operations)
        y = x2 + x2

        assert (x2 == sy.FloatTensor([1, 2, 3, 4, 5])).all()
        assert (y == sy.FloatTensor([2, 4, 6, 8, 10])).all()

        assert x2.id not in me._objects

        x3 = sy.FloatTensor.deser(xs)

        assert x2.id in me._objects


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

<<<<<<< HEAD
    # def test_send_and_get_tensor(self):
    #
    #     x = sy.FloatTensor([1, 2, 3, 4, 5])
    #
    #     xid = x.id
    #
    #     x.send(bob, ptr_id=1234)
    #
    #     # the id of x should have changed to that of the pointer
    #     assert x.id == 1234
    #
    #     # make sure x is not localy registered
    #     assert xid not in me._objects
    #
    #     # make sure x is registered at bob
    #     assert xid in bob._objects
    #
    #     # getting tensor back... putting result into x2
    #     # to show that it should have updated x independently
    #     x2 = x.get()
    #
    #     # make sure x changes id back to what it should
    #     assert x.id == xid
    #
    #     # make sure x is now registered locally
    #     assert xid in me._objects
    #
    #     # make sure x is not registered with bob
    #     assert xid not in bob._objects
    #
    #     # make sure pointer is no longer registered locally
    #     assert 1234 not in me._objects
=======
    def test_send_and_get_tensor(self):

        x = sy.FloatTensor([1, 2, 3, 4, 5])

        xid = x.id

        x.send(bob, ptr_id=1234)

        # the id of x should have changed to that of the pointer
        assert x.id == 1234

        # make sure x is not localy registered
        assert xid not in me._objects

        # make sure x is registered at bob
        assert xid in bob._objects

        # getting tensor back... putting result into x2
        # to show that it should have updated x independently
        x2 = x.get()

        # make sure x changes id back to what it should
        assert x.id == xid

        # make sure x is now registered locally
        assert xid in me._objects

        # make sure x is not registered with bob
        assert xid not in bob._objects

        # make sure pointer is no longer registered locally
        assert 1234 not in me._objects

>>>>>>> 53d30afadee5609733fafe7783631647ad15447c
