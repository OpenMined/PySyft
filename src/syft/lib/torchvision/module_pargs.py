# stdlib
from typing import Any
from typing import Dict
from typing import List

module_pargs: Dict[str, List[Any]] = {}

# For all modules requiring positional arguments to access attributes
module_pargs["torchvision.transforms.Normalize"] = [(1, 2, 3), (1, 2, 3)]
