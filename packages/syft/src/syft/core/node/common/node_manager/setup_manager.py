# stdlib
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

# third party
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

# relative
from ...common.exceptions import SetupNotFoundError
from ..node_table.pdf import PDFObject
from ..node_table.setup import NoSQLSetup
from ..node_table.setup import SetupConfig

# from ..exceptions import SetupNotFoundError
from .database_manager import DatabaseManager
from .database_manager import NoSQLDatabaseManager


class SetupManager(DatabaseManager):

    schema = SetupConfig

    def __init__(self, database: Engine) -> None:
        super().__init__(db=database, schema=SetupManager.schema)

    def register_once(self, **kwargs: Any) -> Any:
        """Register a new object into the database.

        Args:
            parameters : List of object parameters.
        Returns:
            object: Database Object
        """
        _obj = self._schema(**kwargs)
        session_local = sessionmaker(autocommit=False, autoflush=False, bind=self.db)()

        with session_local.begin():
            try:
                _obj.id = 1  # force a single row
                session_local.add(_obj)
                session_local.commit()
            except Exception as e:
                raise e

        return kwargs

    @property
    def node_name(self) -> str:
        setup = super().all()[0]
        return setup.domain_name

    @property
    def id(self) -> int:
        setup = super().all()[0]
        return setup.id

    def first(self, **kwargs: Any) -> SetupConfig:
        result = super().first(**kwargs)
        if not result:
            # raise SetupNotFoundError
            raise Exception
        return result

    def query(self, **kwargs: Any) -> List[SetupConfig]:
        results = super().query(**kwargs)
        if len(results) == 0:
            # raise SetupNotFoundError
            raise Exception
        return results

    def update(self, **kwargs: Any) -> None:
        session_local = sessionmaker(autocommit=False, autoflush=False, bind=self.db)()
        if "daa_document" in kwargs.keys():
            _pdf_obj = PDFObject(binary=kwargs["daa_document"])
            session_local.add(_pdf_obj)
            session_local.commit()
            session_local.flush()
            session_local.refresh(_pdf_obj)
            kwargs["daa_document"] = str(_pdf_obj.id)

        session_local.query(self._schema).update(kwargs)
        session_local.commit()
        session_local.close()


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
