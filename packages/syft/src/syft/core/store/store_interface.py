# stdlib
from abc import ABC
from typing import Any
from typing import Iterable
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

# relative
from ...logger import debug
from ...logger import traceback_and_raise
from ..common.storeable_object import AbstractStorableObject
from ..common.uid import UID
from .storeable_object import StorableObject

StoreKey = Union[UID, str, bytes]


class ObjectStore(ABC):
    """
    ObjectStore is the common interface for all the stores that a Node can handle. This should
    provide a dict-like interface on handling data on a worker. Indexing should be always done
    by using UID objects, while de-indexed value should always be a SerizableObject.
    """

    def __sizeof__(self) -> int:
        """
        Method to return the memory size of the object if possible, if not, it will return __len__.

        Returns:
            int: memory size of the store.
        """
        traceback_and_raise(NotImplementedError)

    def __str__(self) -> str:
        """
        Method to print to first 5 entries of the ObjectStore in a database-like format.

        Returns:
            str: header and first 5 entries of the store in a formatted manner.
        """
        traceback_and_raise(NotImplementedError)

    def __len__(self) -> int:
        """
        Method to return the number of elements in the store.

        Returns:
            int: the number of objects in the store.
        """
        traceback_and_raise(NotImplementedError)

    def keys(self) -> Iterable[UID]:
        """
        Method to return all indexes from the store.

        Returns:
            Iterable[UID]: an iterable containing the keys of the store.
        """
        traceback_and_raise(NotImplementedError)

    def values(self) -> Iterable[StorableObject]:
        """
        Method to return all values from the store.

        Returns:
            Iterable[StorableObject]: an iterable containing the keys of the store.
        """
        traceback_and_raise(NotImplementedError)

    def __contains__(self, key: UID) -> bool:
        """
        Method to check if an UID is present in the store.

        Args:
            key (UID): the UID to be searched.

        Returns:
            bool: if the object is present or not.
        """
        traceback_and_raise(NotImplementedError)

    def __getitem__(self, key: StoreKey) -> StorableObject:
        """
        Method to retrieve an object from the store.

        Args:
            key (UID): The key to be searched for in the store.

        Returns:
            StorableObject: The object associated with the input key.

        Raises:
            ValueError: If the key is not present in the store.
        """
        traceback_and_raise(NotImplementedError)

    def __setitem__(self, key: UID, value: StorableObject) -> None:
        """
        Method to store an object in the store. The difference between this and store is that you
        can you use self.store to store objects at a different key than the one specified in
        the StorableObject.

        This updates the value at the input UID if it exists already.

        Args:
            key (UID): the UID at which to store the StorableObject.
            value (StorableObject): the StorableObject to be stored.
        """
        traceback_and_raise(NotImplementedError)

    def __delitem__(self, key: UID) -> None:
        self.delete(key=key)

    def delete(self, key: UID) -> None:
        """
        We should write custom deletion code so we can check if the item exists
        and then ensure deletion is called at a single place and that full __delitem__
        is used where possible instead of the weaker del ref count.
        Also, care needs to be taken to not capture the deleted item in the Exception
        stack trace so it doesn't keep the item around longer.

        Method to remove an object from the store based on its UID.

        Args:
            key (UID): the key at which to delete the object.

        Raises:
            ValueError: if the key is not present in the store.
        """
        traceback_and_raise(NotImplementedError)

    def clear(self) -> None:
        """
        Clears all storage owned by the store.
        """
        traceback_and_raise(NotImplementedError)

    def get_or_none(
        self, key: UID, proxy_only: bool = False
    ) -> Optional[StorableObject]:
        traceback_and_raise(NotImplementedError)

    def check_collision(self, key: UID) -> None:
        traceback_and_raise(NotImplementedError)

    def get(self, key: StoreKey, proxy_only: bool = False) -> StorableObject:
        traceback_and_raise(NotImplementedError)

    def set(self, key: StoreKey, value: StorableObject) -> None:
        traceback_and_raise(NotImplementedError)

    def has_object(self) -> None:
        traceback_and_raise(NotImplementedError)

    def store_object(self) -> None:
        traceback_and_raise(NotImplementedError)

    def delete_object(self) -> None:
        traceback_and_raise(NotImplementedError)

    def get_objects_of_type(self, obj_type: Type) -> Iterable[AbstractStorableObject]:
        traceback_and_raise(NotImplementedError)

    @property
    def icon(self) -> str:
        return "🗃️"

    @property
    def pprint(self) -> str:
        output = f"{self.icon} ({self.class_name}) {self.peek}"
        return output

    @property
    def peek(self) -> str:
        output = "{"
        for k in self.keys():
            output += f"\n  > {k} {k.emoji()} => {self[k].pprint}\n"
        output += "}"
        return output

    def post_init(self) -> None:
        debug(f"> Creating {self.pprint}")

    @property
    def class_name(self) -> str:
        return str(self.__class__.__name__)

    def resolve_proxy_object(self, obj: Any) -> Any:
        traceback_and_raise(NotImplementedError)

    @staticmethod
    def key_to_str_and_uid(key: StoreKey) -> Tuple[str, UID]:
        if isinstance(key, bytes):
            key = key.decode("utf-8")

        if isinstance(key, str):
            key = UID.from_string(key)

        return str(key.value), key
