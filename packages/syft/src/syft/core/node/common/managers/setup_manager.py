# stdlib
from typing import List
from typing import Union

# relative
from ..tables.setup import SetupConfig

# from ..exceptions import SetupNotFoundError
from .database_manager import DatabaseManager


class SetupManager(DatabaseManager):

    schema = SetupConfig

    def __init__(self, database):
        self._schema = SetupManager.schema
        self.db = database

    @property
    def node_name(self):
        setup = super().all()[0]
        return setup.domain_name

    def first(self, **kwargs) -> Union[None, List]:
        result = super().first(**kwargs)
        if not result:
            raise SetupNotFoundError
        return result

    def query(self, **kwargs) -> Union[None, List]:
        results = super().query(**kwargs)
        if len(results) == 0:
            raise SetupNotFoundError
        return results
