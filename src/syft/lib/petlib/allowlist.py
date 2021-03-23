# stdlib
from typing import Dict
from typing import Union

allowlist: Dict[str, Union[str, Dict[str, str]]] = {}  # (path: str, return_type:type)

allowlist["petlib.ec.EcPt"] = "petlib.ec.EcPt"
allowlist["petlib.ec.EcPt.pt"] = "_cffi_backend._CDataBase"
allowlist["petlib.ec.EcGroup"] = "petlib.ec.EcGroup"
allowlist["petlib.bn.Bn"] = "petlib.bn.Bn"
allowlist["petlib.ec.EcPt.group"] = "petlib.ec.EcGroup"
allowlist["petlib.ec.EcPt.__copy__"] = "petlib.ec.EcPt"
allowlist["petlib.ec.EcPt.pt_add"] = "petlib.ec.EcPt"
allowlist["petlib.ec.EcPt.pt_add_inplace"] = "petlib.ec.EcPt"
allowlist["petlib.ec.EcPt.pt_mul"] = "petlib.ec.EcPt"
allowlist["petlib.ec.EcPt.pt_mul_inplace"] = "petlib.ec.EcPt"
allowlist["petlib.ec.EcPt.__rmul__"] = "petlib.ec.EcPt"
