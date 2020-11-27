# stdlib
from typing import Dict
from typing import Union

allowlist: Dict[str, Union[str, Dict[str, str]]] = {}  # (path: str, return_type:type)

allowlist["sympc.protocol.spdz.spdz.mul_parties"] = "sympc.tensor.ShareTensor"
allowlist["sympc.session.Session.przs_generate_random_share"] = "sympc.tensor.ShareTensor"
allowlist["sympc.session.get_generator"] = "torch.Generator"


allowlist["sympc.tensor.ShareTensor"] = "sympc.tensor.ShareTensor"
allowlist["sympc.tensor.ShareTensor.__add__"] = "sympc.tensor.ShareTensor"
allowlist["sympc.tensor.ShareTensor.__sub__"] = "sympc.tensor.ShareTensor"
allowlist["sympc.tensor.ShareTensor.__mul__"] = "sympc.tensor.ShareTensor"
