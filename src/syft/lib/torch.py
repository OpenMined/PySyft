from ..ast import Globals

import torch

whitelist = {}  # (path: str, return_type:type)
whitelist["torch.tensor"] = "torch.Tensor"
whitelist["torch.Tensor"] = "torch.Tensor"
whitelist["torch.Tensor.__add__"] = "torch.Tensor"
whitelist["torch.zeros"] = "torch.Tensor"
whitelist["torch.ones"] = "torch.Tensor"
whitelist["torch.nn.Linear"] = "torch.nn.Linear"
# whitelist.add("torch.nn.Linear.parameters")


def create_torch_ast():
    ast = Globals()

    for method, return_type_name in whitelist.items():
        ast.add_path(
            path=method, framework_reference=torch, return_type_name=return_type_name
        )

    for klass in ast.classes:
        klass.create_pointer_class()
        klass.create_send_method()
    return ast
