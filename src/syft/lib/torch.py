from ..ast.globals import Globals

import torch

whitelist = set()
whitelist.add("torch.tensor")
whitelist.add("torch.Tensor")
whitelist.add("torch.Tensor.__add__")
whitelist.add("torch.zeros")
whitelist.add("torch.ones")
whitelist.add("torch.nn.Linear")
whitelist.add("torch.nn.Linear.parameters")

ast = Globals()

for method in whitelist:
    ast.add_path(method, torch)

for klass in ast.classes:
    klass.create_pointer_class()
    klass.create_send_method()
