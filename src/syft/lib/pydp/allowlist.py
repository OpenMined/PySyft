# stdlib
from typing import Dict
from typing import Union

from ..misc.union import UnionGenerator

allowlist: Dict[str, Union[str, Dict[str, str]]] = {}  # (path: str, return_type:type)

# --------------------------------------------------------------------------------------
# SECTION - Tensor methods which are tested
# --------------------------------------------------------------------------------------

# SECTION - The main Classes
allowlist["pydp.algorithms.laplacian.BoundedMean"] = "pydp.algorithms.laplacian._bounded_algorithms.BoundedMean"


# SECTION - BoundedMean allowed functions

allowlist["pydp.algorithms.laplacian.BoundedMean.quick_result"] = UnionGenerator["syft.lib.python.Float", "syft.lib.python.Int"]
allowlist["pydp.algorithms.laplacian.BoundedMean.add_entries"] = "syft.lib.python.None"
allowlist["pydp.algorithms.laplacian.BoundedMean.add_entry"] = "syft.lib.python.None"
allowlist["pydp.algorithms.laplacian.BoundedMean.memory_used"] = "syft.lib.python.Float"
allowlist["pydp.algorithms.laplacian.BoundedMean.noise_confidence_interval"] = "syft.lib.python.Float"
allowlist["pydp.algorithms.laplacian.BoundedMean.privacy_budget_left"] = "syft.lib.python.Float"
allowlist["pydp.algorithms.laplacian.BoundedMean.reset"] = "syft.lib.python.None"
allowlist["pydp.algorithms.laplacian.BoundedMean.result"] = UnionGenerator["syft.lib.python.Float", "syft.lib.python.Int"]





