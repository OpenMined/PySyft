# stdlib
from typing import Any
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
            _obj = NoSQLSetup(**kwargs)
            self.add(_obj)
            return _obj

        return None

    @property
    def node_name(self) -> str:
        setup = super().all()[0]
        return setup.domain_name

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

    def update_config(self, **kwargs: Any) -> None:
        setup: NoSQLSetup = self.all()[0]
        self.update(
            search_params={"node_uid": setup.node_uid},
            updated_args=kwargs,
        )
