# third party
import pandas

# relative
from .arrow import numpy_deserialize
from .arrow import numpy_serialize
from .recursive import recursive_serde_register

recursive_serde_register(
    pandas.core.indexes.numeric.Int64Index,
    serialize=lambda x: numpy_serialize(x.values),
    deserialize=lambda buffer: pandas.core.indexes.numeric.Int64Index(
        numpy_deserialize(buffer)
    ),
)

# how else do you import a relative file to execute it?
NOTHING = None
