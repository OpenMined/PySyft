from unittest import TestCase
import syft as sy
import torch
import random

hook = sy.TorchHook()

me = hook.local_worker
me.is_client_worker = False

bob = sy.VirtualWorker(id="bob",hook=hook, is_client_worker=False)
alice = sy.VirtualWorker(id="alice",hook=hook, is_client_worker=False)

bob.add_workers([me, alice])
alice.add_workers([me, bob])
me.add_workers([bob, alice])

torch.manual_seed(1)
random.seed(1)

class TestTensorPointerSerde(TestCase):

    def test_floattensordata2json2floattensordata(self):
        # this tests the serialization / deserialization of the data FloatTensor
        # objects (which is different from those which are merely wrappers).

        x = sy.FloatTensor([1, 2, 3, 4, 5])

        xs = {
                '__FloatTensor__': {
                    'type': 'syft.core.frameworks.torch.tensor.FloatTensor',
                    'torch_type': 'syft.FloatTensor',
                    'data': [1.0, 2.0, 3.0, 4.0, 5.0],
                    'child': {
                        '___LocalTensor__': {
                            'owner': 0,
                            'id': x.id,
                            'torch_type': 'syft.FloatTensor'
            }}}}

        assert x.ser(private=False) == xs

        x2 = sy.FloatTensor.deser(xs, worker=me, acquire=True)

        # ensure values are the same as what was serialized
        assert x2.tolist() == x.tolist()

        # assert the objects are the same
        assert (x == x2).all()

    def test_localtensor2json2localtensor(self):
        xs = {
            '__FloatTensor__': {
                'type': 'syft.core.frameworks.torch.tensor.FloatTensor',
                'torch_type': 'syft.FloatTensor',
                'data': [1.0, 2.0, 3.0, 4.0],
                'child': {
                    '___LocalTensor__': {
                        'owner': 0,
                        'id': 1000,
                        'torch_type': 'syft.FloatTensor'
                    }}}}

        x = sy.FloatTensor([1, 2, 3, 4])
        x.child.id = xs['__FloatTensor__']['child']['___LocalTensor__']['id']

        # check that serialization is correct
        assert xs == x.ser(private=False)

        # reset ID for further testing
        xs['__FloatTensor__']['child']['___LocalTensor__']['id'] = 54321

        x = sy.FloatTensor.deser(xs, worker=me, acquire=True)

        # correct id
        assert x.id == xs['__FloatTensor__']['child']['___LocalTensor__']['id']

        # correct owner
        assert x.owner.id == xs['__FloatTensor__']['child']['___LocalTensor__']['owner']

        # correct type
        assert type(x).__name__ == xs['__FloatTensor__']['type'].split(".")[-1]

        # correct size
        assert len(x) == 4

        # correct data
        assert (x[0:4] == sy.FloatTensor([1, 2, 3, 4])).all()

        # object shouldn't be in registry yet
        assert x.id not in me._objects

    def test_floattensor2json2floattensor(self):

        xs = {
            '__FloatTensor__': {
                'type': 'syft.core.frameworks.torch.tensor.FloatTensor',
                'torch_type': 'syft.FloatTensor',
                'data': [1.0, 2.0, 3.0, 4.0, 5.0],
                'child': {
                    '___LocalTensor__': {
                        'owner': 0,
                        'id': 234152,
                        'torch_type': 'syft.FloatTensor'
                    }}}}

        x = sy.FloatTensor([1, 2, 3, 4, 5])
        x.child.id = 234152

        # test that serialization happens correctly
        assert x.ser(private=False) == xs

        # initialize tensor without registering it
        x2 = sy.FloatTensor.deser(xs, worker=me, acquire=True)

        # check id and owner are correct
        assert x2.id == 234152
        assert x2.owner.id == 0

        # make sure it works (can do operations)
        y = x2 + x2

        assert (x2 == sy.FloatTensor([1, 2, 3, 4, 5])).all()
        assert (y == sy.FloatTensor([2, 4, 6, 8, 10])).all()

        assert x2.id not in me._objects

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

        # ensure that it was registered
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


    def test_send_and_get_tensor(self):

        x = sy.FloatTensor([1, 2, 3, 4, 5])

        xid = x.id

        x.send(bob, ptr_id=1234)

        # getting tensor back and putting result into x2
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


if __name__ == '__main__':
    unittest.main()