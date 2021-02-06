# stdlib
from typing import Any
from typing import Callable as CallableT
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

# syft relative
from .. import ast
from ..logger import traceback_and_raise


class Property(ast.attribute.Attribute):
    def __init__(
        self,
        path_and_name: str,
        object_ref: Optional[Any] = None,
        return_type_name: Optional[str] = None,
        require_pargs: bool = False,
        parg_list: List[Any] = [],
        client: Optional[Any] = None,
    ):
        super().__init__(
            path_and_name=path_and_name,
            object_ref=object_ref,
            return_type_name=return_type_name,
            require_pargs=require_pargs,
            parg_list=parg_list,
            client=client,
        )

        self.is_static = False

    def __call__(
        self,
        *args: Tuple[Any, ...],
        **kwargs: Any,
    ) -> Optional[Union[Any, CallableT]]:
        traceback_and_raise(ValueError("Property should never be called."))
