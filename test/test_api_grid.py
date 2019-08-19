import pytest
import requests
import grid as gr
import unittest
import json
import syft as sy
import torch as th
from test import PORTS, IDS, GATEWAY_URL

hook = sy.TorchHook(th)


class GridAPITest(unittest.TestCase):
    def setUp(self):
        self.my_grid = gr.GridNetwork(GATEWAY_URL)

    def tearDown(self):
        self.my_grid.disconnect_nodes()

    def test_connected_nodes(self):
        response = json.loads(requests.get(GATEWAY_URL + "/connected-nodes").content)
        self.assertEqual(len(response["grid-nodes"]), 3)
        self.assertTrue("Bob" in response["grid-nodes"])
        self.assertTrue("Alice" in response["grid-nodes"])
        self.assertTrue("James" in response["grid-nodes"])

    def test_grid_search(self):
        nodes = []
        for (node_id, port) in zip(IDS, PORTS):
            node = gr.WebsocketGridClient(
                hook, "http://localhost:" + port + "/", id=node_id
            )
            node.connect()
            nodes.append(node)

        x = th.tensor([1, 2, 3, 4, 5]).tag("#simple-tensor").describe("Simple tensor")
        y = (
            th.tensor([[4], [5], [7], [8]])
            .tag("#2d-tensor")
            .describe("2d tensor example")
        )
        z = (
            th.tensor([[0, 0, 0, 0, 0]])
            .tag("#zeros-tensor")
            .describe("tensor with zeros")
        )
        w = (
            th.tensor([[0, 0, 0, 0, 0]])
            .tag("#zeros-tensor")
            .describe("tensor with zeros")
        )

        x_s = x.send(nodes[0])
        y_s = y.send(nodes[1])
        z_s = z.send(nodes[2])
        w_s = w.send(nodes[0])

        x_s.child.garbage_collect_data = False
        y_s.child.garbage_collect_data = False
        z_s.child.garbage_collect_data = False
        w_s.child.garbage_collect_data = False

        for node in nodes:
            node.disconnect()

        simple_tensor = self.my_grid.search("#simple-tensor")
        self.assertEqual(len(simple_tensor), 1)
        zeros_tensor = self.my_grid.search("#zeros-tensor")
        self.assertEqual(len(zeros_tensor), 2)
        d_tensor = self.my_grid.search("#2d-tensor")
        self.assertEqual(len(d_tensor), 1)
        nothing = self.my_grid.search("#nothing")
        self.assertEqual(len(nothing), 0)
