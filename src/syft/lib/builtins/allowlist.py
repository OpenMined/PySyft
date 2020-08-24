from typing import Dict

allowlist: Dict[str, str] = {}  # (path: str, return_type:type)

# SECTION - add the capital constructor
allowlist["builtins.int"] = "builtins.int"