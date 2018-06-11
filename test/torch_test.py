from unittest import TestCase
from syft.core.hooks import TorchHook
from syft.core.hooks import torch
from syft.core.workers import VirtualWorker

import json

class TestTorchOverride(TestCase):
    def test___repr__(self):

        hook = TorchHook(verbose=False)
        x = torch.FloatTensor([1,2,3,4,5])
        assert x.__repr__() == '\n 1\n 2\n 3\n 4\n 5\n[torch.FloatTensor of size 5]\n'

    def test_send_(self):

        hook = TorchHook(verbose=False)
        local = hook.local_worker
        remote = VirtualWorker(id=1, hook=hook)

        x = torch.FloatTensor([1,2,3,4,5])
        x = x.send_(remote)
        assert x.id in remote._objects


    def test_deser(self):

        unregistered_tensor = torch.FloatTensor.deser(torch.FloatTensor,[1.0, 2.0, 3.0, 4.0, 5.0])
        assert (unregistered_tensor == torch.FloatTensor([1,2,3,4,5])).float().sum() == 5

    def test_deser_from_message(self):

        hook = TorchHook(verbose=False)

        message_obj = json.loads(' {"torch_type": "torch.FloatTensor", "data": [1.0, 2.0, 3.0, 4.0, 5.0], "id": 9756847736, "owners": [1], "is_pointer": false}')
        obj_type = hook.types_guard(message_obj['torch_type'])
        unregistered_tensor = torch.FloatTensor.deser(obj_type,message_obj['data'])
        
        assert (unregistered_tensor == torch.FloatTensor([1,2,3,4,5])).float().sum() == 5

        # has not been registered
        assert unregistered_tensor.id != 9756847736



