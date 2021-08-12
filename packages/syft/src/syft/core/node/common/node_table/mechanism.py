# stdlib
from typing import Any

# third party
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import LargeBinary

# syft absolute
from syft import deserialize
from syft import serialize
from syft.core.adp.idp_gaussian_mechanism import iDPGaussianMechanism

# relative
from . import Base


class Mechanism(Base):
    __tablename__ = "mechanism"

    id = Column(Integer(), primary_key=True, autoincrement=True)
    mechanism_bin = Column(LargeBinary(3072), default=None)

    @property
    def obj(self) -> Any:
        obj_list = deserialize(
            self.mechanism_bin, from_bytes=True
        )  # TODO: techdebt fix
        # iDPGaussianMechanism.__new__ used by the recursive serde
        # will not initialize iDPGaussianMechanism super class.
        # Since we're extending a third party lib class to perform dp (autodp lib)
        # we need to force super class initialization in order to provide support for
        # autodp internal methods created in execution time (e.g. RenyiDP)
        if len(obj_list):
            _ = [
                iDPGaussianMechanism.__init__(
                    obj_list[i],
                    obj_list[i].params["sigma"],
                    obj_list[i].params["value"],
                    obj_list[i].params["L"],
                    obj_list[i].entity,
                    user_key=obj_list[i].user_key,
                )
                for i in range(len(obj_list))
            ]
        return obj_list

    @obj.setter
    def obj(self, value: Any) -> None:
        self.mechanism_bin = serialize(value, to_bytes=True)  # TODO: techdebt fix
