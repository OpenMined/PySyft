# stdlib
from typing import List
from typing import Union

# relative
from ..exceptions import EnvironmentNotFoundError
from ..node_table.environment import Environment
from ..node_table.user_environment import UserEnvironment
from .database_manager import DatabaseManager


class EnvironmentManager(DatabaseManager):

    schema = Environment
    user_env_association_schema = UserEnvironment

    def __init__(self, database):
        super().__init__(schema=EnvironmentManager.schema, db=database)
        self._association_schema = EnvironmentManager.user_env_association_schema

    def association(self, user_id: str, env_id: str):
        new_association_obj = self._association_schema(user=user_id, environment=env_id)
        self.db.session.add(new_association_obj)
        self.db.session.commit()

    def get_environments(self, **kwargs):
        objects = (
            self.db.session.query(self._association_schema).filter_by(**kwargs).all()
        )
        return objects

    def get_all_associations(self):
        return list(self.db.session.query(self._association_schema).all())

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
