from .ast import Globals

import torch

whitelist = set()
whitelist.add('torch.tensor')
whitelist.add('torch.Tensor')
whitelist.add('torch.Tensor.__add__')
whitelist.add('torch.zeros')
whitelist.add('torch.ones')
whitelist.add('torch.nn.Linear')
whitelist.add('torch.nn.Linear.parameters')

torch_ast = Globals()

for method in whitelist:
    torch_ast.add_path(method, torch)

