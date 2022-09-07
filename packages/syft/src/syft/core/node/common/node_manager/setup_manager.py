# stdlib
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

# relative
from ...common.exceptions import SetupNotFoundError
from ..node_table.setup import NoSQLSetup
from .database_manager import NoSQLDatabaseManager


class NoSQLSetupManager(NoSQLDatabaseManager):
    """Class to manage user database actions."""

    _collection_name = "setup"
    __canonical_object_name__ = "Setup"

    def register_once(self, **kwargs: Any) -> Optional[NoSQLSetup]:
        """Register a new object into the database.

        Args:
            parameters : List of object parameters.
        Returns:
            object: Database Object
        """
        curr_len = len(self)
        if curr_len == 0:
            _obj = NoSQLSetup(id_int=1, **kwargs)
            self.add(_obj)
            return _obj

        return None

    @property
    def node_name(self) -> str:
        setup = super().all()[0]
        return setup.domain_name

    @property
    def id(self) -> int:
        setup = super().all()[0]
        return setup.id_int

    def first(self, **kwargs: Any) -> NoSQLSetup:
        result = super().all()
        if not result:
            raise SetupNotFoundError
        return result[0]

    def query(self, **kwargs: Any) -> List[NoSQLSetup]:
        results = super().query(**kwargs)
        if len(results) == 0:
            raise SetupNotFoundError
        return results

    def update(self, **kwargs: Any) -> None:
        setup: NoSQLSetup = self.all()[0]
        attributes: Dict[str, Any] = {}
        for k, v in kwargs.items():
            if k not in setup.__attr_state__:
                raise ValueError(f"Cannot set an non existing field:{k} to Node")
            else:
                setattr(setup, k, v)
            if k in setup.__attr_searchable__:
                attributes[k] = v
        attributes["__blob__"] = setup.to_bytes()

        self.update_one(query={"id_int": setup.id_int}, values=attributes)
