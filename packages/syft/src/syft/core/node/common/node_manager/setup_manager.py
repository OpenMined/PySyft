# stdlib
from typing import Any
from typing import List

# third party
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

# relative
from ..node_table.pdf import PDFObject
from ..node_table.setup import SetupConfig

# from ..exceptions import SetupNotFoundError
from .database_manager import DatabaseManager


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
            nrows = len(self)
            if nrows == 0:
                session_local.add(_obj)
                session_local.commit()

        return _obj

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
