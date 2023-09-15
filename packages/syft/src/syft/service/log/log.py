from ...serde.serializable import serializable
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ...types.uid import UID

@serializable()
class SyftLog(SyftObject):
    __canonical_name__ = "SyftLog"
    __version__ = SYFT_OBJECT_VERSION_1
    
    stdout: str = ""
    
    def append(self, new_str: str) -> None:
        self.stdout += new_str

class SyftLogger():
    def __init__(self, context, log_id) -> None:
        self.context = context
        self.log_id = log_id
        
    def print(self, *args, sep=None, end=None, file=None, flush=None):
        import sys
        print("USED SYFT LOGGER", file=sys.stderr)
        if sep is None:
            sep=" "
        if end is None:
            end = '\n'
        new_str = sep.join(args) + end
        log_service = self.context.node.get_service("LogService")
        log_service.append(context=self.context, uid=self.log_id, new_str=new_str)
        