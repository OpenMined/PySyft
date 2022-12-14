# stdlib
from typing import List as TypeList

# third party
from typing_extensions import final

# relative
from .....common.message import ImmediateSyftMessageWithoutReply  # type: ignore
from .....common.serde.serializable import serializable  # type: ignore
from .....common.uid import UID  # type: ignore
from .....io.address import Address  # type: ignore


@serializable(recursive_serde=True)
@final
class PublishScalarsAction(ImmediateSyftMessageWithoutReply):
    __attr_allowlist__ = (
        "id_at_location",
        "publish_ids_at_location",
        "sigma",
        "private",
        "address",
        "_id",
    )

    def __init__(
        self,
        id_at_location: UID,
        address: Address,
        publish_ids_at_location: TypeList[UID],
        sigma: float,
        private: bool,
    ):
        super().__init__(address=address)
        self.id_at_location = id_at_location
        self.publish_ids_at_location = publish_ids_at_location
        self.sigma = sigma
        self.private = private
