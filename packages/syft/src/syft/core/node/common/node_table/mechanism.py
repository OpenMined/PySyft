# stdlib
from typing import Any

# third party
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import LargeBinary
from sqlalchemy import String

# relative
from . import Base
from ..... import deserialize
from ..... import serialize
from ....adp.idp_gaussian_mechanism import iDPGaussianMechanism


class Mechanism(Base):
    __tablename__ = "mechanism"

    id = Column(Integer(), primary_key=True, autoincrement=True)

    # "Kritika the data scientist"
    user_key = Column(LargeBinary(2048), default="")

    # the name of the medical patient whose private information was queried
    entity_name = Column(String(1024), default="")

    # list of mechanisms about medical patients = [Bob, Sue, James]
    mechanism_bin = Column(LargeBinary(3072), default=None)

    @property
    def obj(self) -> Any:
        _obj = deserialize(self.mechanism_bin, from_bytes=True)  # TODO: techdebt fix
        # iDPGaussianMechanism.__new__ used by the recursive serde
        # will not initialize iDPGaussianMechanism super class.
        # Since we're extending a third party lib class to perform dp (autodp lib)
        # we need to force super class initialization in order to provide support for
        # autodp internal methods created in execution time (e.g. RenyiDP)
        # if len(obj_list):
        iDPGaussianMechanism.__init__(
            self=_obj,
            sigma=_obj.params["sigma"],
            squared_l2_norm=_obj.params["private_value"],
            squared_l2_norm_upper_bound=_obj.params["public_value"],
            L=_obj.params["L"],
            entity_name=_obj.entity_name,
            user_key=_obj.user_key,
        )

        return _obj

    @obj.setter
    def obj(self, value: Any) -> None:
        self.mechanism_bin = serialize(value, to_bytes=True)  # TODO: techdebt fix
