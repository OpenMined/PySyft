# stdlib
from typing import Any
from typing import Callable
from typing import Literal
from typing import Sequence
from typing import Tuple
from typing import Union
from unittest import mock

# third party
from bson.objectid import ObjectId
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import CollectionInvalid
from pymongo.errors import DuplicateKeyError
from pymongo.errors import InvalidName
from pymongo.errors import OperationFailure

def patch(
    servers: Union[str, Tuple[str, int], Sequence[Union[str, Tuple[str, int]]]] = ...,
    on_new: Literal["error", "create", "timeout", "pymongo"] = ...,
) -> mock._patch: ...

_FeatureName = Literal["collation", "session"]

def ignore_feature(feature: _FeatureName) -> None: ...
def warn_on_feature(feature: _FeatureName) -> None: ...

SERVER_VERSION: str = ...
