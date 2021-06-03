from nacl.signing import VerifyKey
from typing import List, Dict
from syft.core.common.uid import UID


class ResultPermission():

    def __init__(self, id: UID, verify_key: VerifyKey, method_name: str, args: List[UID]=None, kwargs: Dict[str, UID]=None):
        self.id=id
        self.verify_key=verify_key
        self.method_name=method_name
        self.args=args if args is not None else []
        self.kwargs=kwargs if kwargs is not None else dict()

    def __eq__(self, o: 'ResultPermission') -> bool:
        return self.id == o.id and self.verify_key == o.verify_key and self.method_name == o.method_name and self.args == o.args and self.kwargs ==o.kwargs
    
    def matches(self, id, verify_key, method_name, args, kwargs):
        return self.id==id and self.verify_key == verify_key and self.method_name==method_name and self.args == args and self.kwargs == kwargs
    