# stdlib
from typing import Any
from typing import Dict
from typing import List
from typing import Union

allowlist: Dict[str, Union[str, Dict[str, str]]] = {}  # (path: str, return_type:type)
module_pargs: Dict[str, List[Any]] = {}

# MNIST
allowlist["torchvision.transforms.Compose"] = "torchvision.transforms.Compose"
# allowlist["torchvision.transforms.Compose.__iter__"] = "torchvision.transforms.ToTensor"
# TODO: Compose.transforms property only exists on the object not on the class?
# allowlist["torchvision.transforms.Compose.transforms"] = "syft.lib.python.List"
allowlist["torchvision.transforms.ToTensor"] = "torchvision.transforms.ToTensor"
allowlist["torchvision.transforms.Normalize"] = "torchvision.transforms.Normalize"
# TODO: Normalize properties only exists on the object not on the class?
allowlist["torchvision.transforms.Normalize.inplace"] = "syft.lib.python.Bool"
# TODO: mean and std are actually tuples
allowlist["torchvision.transforms.Normalize.mean"] = "syft.lib.python.List"
allowlist["torchvision.transforms.Normalize.std"] = "syft.lib.python.List"
allowlist["torchvision.datasets.MNIST"] = "torchvision.datasets.MNIST"
allowlist["torchvision.datasets.MNIST.__len__"] = "syft.lib.python.Int"
allowlist["torchvision.datasets.VisionDataset"] = "torchvision.datasets.VisionDataset"
allowlist["torchvision.datasets.VisionDataset.__len__"] = "syft.lib.python.Int"
# For all modules requiring positional arguments to access attributes
module_pargs["torchvision.transforms.Normalize"] = [(1, 2, 3), (1, 2, 3)]
