import random

from syft.frameworks.torch.tensors import PointerTensor
from syft.frameworks.torch.hook import TorchHook
from syft.workers.virtual import VirtualWorker
from unittest import TestCase
import torch


class TestTorchTensor(TestCase):
    def setUp(self):
        self.hook = TorchHook(torch)
        self.me = self.hook.local_worker
        self.bob = VirtualWorker()
