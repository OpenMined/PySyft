from dataclasses import dataclass
from typing import Dict, Callable


@dataclass
class SerializationStore:
    type_to_schema: Dict[type, Dict[int, Callable]]
    schema_to_type: Dict[int, Dict[type, Callable]]
