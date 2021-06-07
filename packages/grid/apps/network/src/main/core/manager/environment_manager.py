# stdlib
from datetime import datetime
from typing import List
from typing import Union

# grid relative
from ..database.environment.environment import Environment
from ..database.environment.environment import states
from ..database.environment.user_environment import UserEnvironment
from ..exceptions import EnvironmentNotFoundError
from .database_manager import DatabaseManager


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

    def delete_associations(self, environment_id):
        # Delete User environment Association
        associations = (
            self.db.session.query(self._association_schema)
            .filter_by(environment=environment_id)
            .all()
        )
        for association in associations:
            self.db.session.delete(association)

        self.db.session.commit()

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

    def set(self, id, **kwargs):
        self.modify({"id": id}, {**kwargs})
