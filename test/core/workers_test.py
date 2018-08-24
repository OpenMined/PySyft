from unittest import TestCase

import torch

from syft.core.hooks import TorchHook


class TestBaseWorker(TestCase):
    def test_get_obj(self):
        hook = TorchHook()
        local = hook.local_worker
        objects = [torch.FloatTensor(), torch.FloatTensor(), torch.FloatTensor()]
        local._objects = {int(1): objects[0], 'two': objects[1], float(3.0): objects[2]}
        self.assertIs(local.get_obj(int(1)), objects[0])
        self.assertIs(local.get_obj('two'), objects[1])
        with self.assertRaises(TypeError):
            local.get_obj(float(3.0))
