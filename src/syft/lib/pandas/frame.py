# stdlib
from typing import List
from typing import Optional

# third party
import pandas as pd
from google.protobuf.reflection import GeneratedProtocolMessageType

# syft relative
from ...core.common.uid import UID
from ...core.store.storeable_object import StorableObject
from ...proto.lib.pandas.frame_pb2 import PandasDataFrame as PandasDataFrame_PB
from ...util import aggressive_set_attr
from .frame_util import protobuf_dataframe_deserializer
from .frame_util import protobuf_dataframe_serializer

pd_frame_type = type(pd.DataFrame())
pd_series_type = type(pd.Series())


class PandasDataFrameWrapper(StorableObject):
    def __init__(self, value: object):
        super().__init__(
            data=value,
            id=getattr(value, "id", UID()),
            tags=getattr(value, "tags", []),
            description=getattr(value, "description", ""),
        )
        self.value = value

    def _data_object2proto(self) -> PandasDataFrame_PB:
        proto = protobuf_dataframe_serializer(self.value)
        return proto

    @staticmethod
    def _data_proto2object(proto: PandasDataFrame_PB) -> pd.DataFrame:
        df = protobuf_dataframe_deserializer(proto)
        return df

    @staticmethod
    def get_data_protobuf_schema() -> GeneratedProtocolMessageType:
        return PandasDataFrame_PB

    @staticmethod
    def get_wrapped_type() -> type:
        return pd_frame_type

    @staticmethod
    def construct_new_object(
        id: UID,
        data: StorableObject,
        description: Optional[str],
        tags: Optional[List[str]],
    ) -> StorableObject:
        data.id = id
        data.tags = tags
        data.description = description
        return data


aggressive_set_attr(
    obj=pd_frame_type, name="serializable_wrapper_type", attr=PandasDataFrameWrapper
)
