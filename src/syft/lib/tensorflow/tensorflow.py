# from ..ast.globals import Globals
#
# import tensorflow
#
# allowlist = {} # (path: str, return_type:type)
# allowlist["torch.tensor"] = "torch.Tensor"
# allowlist["torch.Tensor"] = "torch.Tensor"
# allowlist["torch.Tensor.__add__"] = "torch.Tensor"
# allowlist["torch.Tensor.__sub__"] = "torch.Tensor"
# allowlist["torch.zeros"] = "torch.Tensor"
# allowlist["torch.ones"] = "torch.Tensor"
# allowlist["torch.nn.Linear"] = "torch.nn.Linear"
# # allowlist.add("torch.nn.Linear.parameters")
#
# ast = Globals()
#
# for method, return_type_name in allowlist.items():
#     ast.add_path(path=method, framework_reference=tensorflow, return_type_name=return_type_name)
#
# for klass in ast.classes:
#     klass.create_pointer_class()
#     klass.create_send_method()
#
