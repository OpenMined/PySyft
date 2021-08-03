# stdlib
from typing import Any
from typing import List

# third party
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Query

# relative
from ..exceptions import EnvironmentNotFoundError
from ..node_table.environment import Environment
from ..node_table.user_environment import UserEnvironment
from .database_manager import DatabaseManager


class EnvironmentManager(DatabaseManager):

    schema = Environment
    user_env_association_schema = UserEnvironment

    def __init__(self, database: Engine) -> None:
        super().__init__(schema=EnvironmentManager.schema, db=database)
        self._association_schema = EnvironmentManager.user_env_association_schema

    def association(self, user_id: str, env_id: str) -> None:
        new_association_obj = self._association_schema(user=user_id, environment=env_id)
        self.db.session.add(new_association_obj)
        self.db.session.commit()

    def get_environments(self, **kwargs: Any) -> List[Environment]:
        objects = (
            self.db.session.query(self._association_schema).filter_by(**kwargs).all()
        )
        return objects

    def get_all_associations(self) -> List[UserEnvironment]:
        return list(self.db.session.query(self._association_schema).all())

    def delete_associations(self, environment_id: int) -> None:
        # Delete User environment Association
        associations = (
            self.db.session.query(self._association_schema)
            .filter_by(environment=environment_id)
            .all()
        )
        for association in associations:
            self.db.session.delete(association)

        self.db.session.commit()

    def first(self, **kwargs: Any) -> Environment:
        result = super().first(**kwargs)
        if not result:
            raise EnvironmentNotFoundError
        return result

    def query(self, **kwargs: Any) -> Query:
        results = super().query(**kwargs)
        return results

    def set(self, id: int, **kwargs: Any) -> None:
        self.modify({"id": id}, {**kwargs})
