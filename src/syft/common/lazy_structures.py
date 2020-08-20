from typing import Callable, Any, Dict, Set


class LazyDict:
    """
    Sometimes pre-populating a dictionary with every item it needs to serve
    to fulfill a piece of functionality is very expensive. A LazyDict lets
    us only add objects to the dictionary (or decide *whether* add them)
    when they are requested. After they have been requested the dictionary
    continues to hold onto the object for later use. In this way, the
    dictionary is really just a cache of objects. And if an object is never
    requested, then we never expend any cycles adding it to the dictionary.

    The overall behavior is:
    * if the element is found, return the element.
    * else, update the elements of the dict in a lazy manner.
    * retry the search
        - if the element is still not found - return False
        - if the element is found - return True

    Attributes:
         _dict: internal dict to store the elements of the lazy dict.

    """

    __slots__ = ["_update_rule", "_dict"]

    def __init__(self, update_rule: Callable):
        self._dict: Dict[Any, Any] = {}
        self._update_rule = update_rule

    def __sizeof__(self) -> int:
        """
        Method that returns the size of the wrapped dict.

        Returns:
              int: size of the original dict.
        """
        return self._dict.__sizeof__()

    def __len__(self) -> int:
        """
        Method that returns the size of the wrapped dict.

        Returns:
            int: length of the original dict.
        """
        return len(self._dict)

    def keys(self) -> Any:
        """
        Method that returns the keys used in the wrapped dict.

        Returns:
            Any: the keys used for indexing.
        """
        return self._dict.keys()

    def values(self) -> Any:
        """
        Method that returns the values stored in the wrapped dict.

        Returns:
            Any: they values stored in the dict.
        """
        return self._dict.values()

    def __contains__(self, item: Any) -> bool:
        """
        Method that checks if an object is being used as a key in the wrapped
        dict.

        Args:
            item (Any): the key to be searched for.

        Returns:
            bool: if the object is present or not.
        """
        contains = item in self._dict
        if not contains:
            self._update_rule(self, item)
            contains = item in self._dict
        return contains

    def __setitem__(self, key: Any, value: Any) -> None:
        """
        Method that sets an object at a given key in the wrapped dict.

        Args:
              key (Any): the key to be used in the dict.
              value (Any): the value to be used in the dict.
        """
        self._dict[key] = value

    def __getitem__(self, item: Any) -> Any:
        if item not in self._dict:
            self._update_rule()
        return self._dict[item]

    def __str__(self) -> str:
        return str(self._dict)

    def __repr__(self) -> str:
        return repr(self._dict)

    def clear(self) -> None:
        self._dict.clear()


class LazySet:
    """
    Struct that simulates the behavior of a normal set, but that a
    fallback update method when an object is not found to update the
    set.

   The overall behavior is:
    * if the element is found, return the element.
    * else, update the elements of the dict in a lazy manner.
    * retry the search
        - if the element is still not found - return False
        - if the element is found - return True

    Attributes:
         _set: internal set to store the elements of the lazy set.

    """

    __slots__ = ["_update_rule", "_set"]

    def __init__(self, update_rule: Callable):
        self._set: Set[Any] = set()
        self._update_rule = update_rule

    def __sizeof__(self) -> int:
        """
        Method that returns the size of the wrapped set.

        Returns:
              int: size of the original set.
        """
        return self._set.__sizeof__()

    def __len__(self) -> int:
        """
        Method that returns the size of the wrapped set.

        Returns:
            int: length of the original set.
        """
        return len(self._set)

    def __contains__(self, item: Any) -> bool:
        """
        Method that checks if an object is inside the wrapped set.

        Args:
            item (Any): the key to be searched for.

        Returns:
            bool: if the object is present or not.
        """
        contains = item in self._set
        if not contains:
            self._update_rule()
            contains = item in self._set
        return contains

    def add(self, obj: Any) -> None:
        self._set.add(obj)

    def clear(self) -> None:
        self._set.clear()

    def __str__(self) -> str:
        return str(self._set)

    def __repr__(self) -> str:
        return repr(self._set)
