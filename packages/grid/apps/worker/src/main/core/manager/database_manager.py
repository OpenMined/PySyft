# stdlib
from typing import Dict, List, Type, Union

from flask_sqlalchemy import SQLAlchemy

from main.core.database import BaseModel, db


class DatabaseManager:
    _schema: BaseModel
    db: SQLAlchemy

    def register(self, **kwargs) -> BaseModel:
        """Register e  new object into the database.

        Args:
            parameters : List of object parameters.
        Returns:
            object: Database Object
        """
        _obj = self._schema(**kwargs)
        self.db.session.add(_obj)
        self.db.session.commit()
        return _obj

    def query(self, **kwargs) -> Union[List[BaseModel], BaseModel]:
        """Query db objects filtering by parameters
        Args:
            parameters : List of parameters used to filter.
        """
        objects = self.db.session.query(self._schema).filter_by(**kwargs).all()
        return objects

    def first(self, **kwargs) -> Union[None, BaseModel]:
        """Query db objects filtering by parameters
        Args:
            parameters : List of parameters used to filter.
        """
        objects = self.db.session.query(self._schema).filter_by(**kwargs).first()
        return objects

    def all(self) -> List[BaseModel]:
        return list(self.db.session.query(self._schema).all())

    def delete(self, **kwargs):
        """Delete an object from the database.

        Args:
            parameters: Parameters used to filter the object.
        """
        object_to_delete = self.query(**kwargs)[0]
        self.db.session.delete(object_to_delete)
        self.db.session.commit()

    def modify(self, query, values):
        """Modifies one or many records."""
        self.db.session.query(self._schema).filter_by(**query).update(values)
        self.db.session.commit()

    def contain(self, **kwargs) -> bool:
        objects = self.db.session.query(self._schema).filter_by(**kwargs).all()
        return len(objects) != 0

    def __len__(self) -> int:
        return self.db.session.query(self._schema).count()
