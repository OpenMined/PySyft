from typing import Any, Tuple, Dict, List, Union


class AbstractStorage():

    def __init__():
        pass

    def register(self, key: Any, value: Any) -> None:
        raise NotImplementedError("This method should be implemented by the child classes")

    def get(self, **kwargs: Dict[Any, Any]) -> Union[ List[Any], None ]:
        raise NotImplementedError("This method should be implemented by the child classes")

    def get(self, *args: List[Any]) -> Union[Any, None]:
        raise NotImplementedError("This method should be implemented by the child classes")
