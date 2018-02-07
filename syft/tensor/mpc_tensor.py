import numpy as np
import syft.controller
from .float_tensor import FloatTensor
from grid.grid import PubSub
from uuid import uuid4


class MPCTensor():
    def __init__(self, data, data_as_tensor=False, id=None, shards=[0], total_shards=1, pubsub=None):
        if data_as_tensor:
            self.data = data
        else:
            self.data = FloatTensor(data)
        if id is not None:
            self.id = id
        else:
            self.tensor_id = uuid4()
        self.shards = shards
        self.total_shards = total_shards
        if pubsub is None:
            self.pubsub = PubSub()
        else:
            self.pubsub = pubsub

    def shard(self):
        rand_tensor = self.data.random_()
        self.data = self.data-rand_tensor
        self.total_shards += 1
        return MPCTensor(rand_tensor, True, self.tensor_id, [self.total_shards-1], self.total_shards, self.pubsub)

    def share(self):
        temp_pubsub = self.pubsub
        self.pubsub = None
        temp_pubsub.publish("testopenmined", self)

    def __add__(self, x):
        if self.shards == x.shards:
            return MPCTensor(self.data+x.data, True, self.id+x.id, self.shards, self.total_shards)
        else:
            raise ValueError("Different Shards")

    def receive(self):
        self.pubsub.listen_to_channel(self.recombine, 'testopenmined')

    def recombine(self, x):
        if self.id == x.id:
            self.data += x.data
            self.shards = self.shards.extend(x.shards)
        else:
            raise ValueError("Id mismatch")
