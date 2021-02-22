from typing import List
from typing import Union
from .database_manager import DatabaseManager
from ..database.environment.environment import Environment
from ..database.environment.user_environment import UserEnvironment
from ..exceptions import (
    EnvironmentNotFoundError,
)


class EnvironmentManager(DatabaseManager):

    schema = Environment
    user_env_association_schema = UserEnvironment

    def __init__(self, database):
        self._schema = EnvironmentManager.schema
        self._association_schema = EnvironmentManager.user_env_association_schema
        self.db = database

    def association(self, user_id: str, env_id: str):
        new_association_obj = self._association_schema(user=user_id, environment=env_id)
        self.db.session.add(new_association_obj)
        self.db.session.commit()

    def get_environments(self, **kwargs):
        objects = (
            self.db.session.query(self._association_schema).filter_by(**kwargs).all()
        )
        return objects

    def first(self, **kwargs) -> Union[None, List]:
        result = super().first(**kwargs)
        if not result:
            raise EnvironmentNotFoundError
        return result

    def query(self, **kwargs) -> Union[None, List]:
        results = super().query(**kwargs)
        if len(results) == 0:
            raise EnvironmentNotFoundError
        return results
