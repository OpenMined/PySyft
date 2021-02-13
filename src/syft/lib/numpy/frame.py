# third party
import numpy as np
import pandas as pd

# syft relative
from ...proto.core.store.store_object_pb2 import StorableObject as StorableObject_PB
from ...util import aggressive_set_attr
from ..pandas.frame import PandasDataFrameWrapper


class NumpyNdarrayWrapper(PandasDataFrameWrapper):
    def __init__(self, value: object):
        super().__init__(value)
        self.value = pd.DataFrame(self.value)

    @staticmethod
    def _proto2object(proto: StorableObject_PB) -> np.ndarray:
        return (
            super(NumpyNdarrayWrapper, NumpyNdarrayWrapper)
            ._data_proto2object(proto)
            .to_numpy()
        )


aggressive_set_attr(
    obj=np.ndarray, name="serializable_wrapper_type", attr=NumpyNdarrayWrapper
)
