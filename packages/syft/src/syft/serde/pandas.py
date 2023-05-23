# stdlib
import io
import json
from typing import Any

# third party
from matplotlib.axes._subplots import Subplot
import numpy
import numpy as np
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


def string_serialize(a: Any) -> str:
    memfile = io.BytesIO()
    numpy.save(memfile, a)
    serialized = memfile.getvalue()
    serialized_as_json = json.dumps(serialized.decode("latin-1"))
    return serialized_as_json


def string_deserialize(serialized_as_json: Any) -> str:
    memfile = io.BytesIO()
    # memfile.write(serialized)
    # Or if you're deserializing from JSON:
    memfile.write(json.loads(serialized_as_json).encode("latin-1"))
    memfile.seek(0)
    a = numpy.load(memfile)
    return a


recursive_serde_register(
    pandas.core.indexes.datetimes.DatetimeIndex,
    serialize=lambda x: string_serialize(x),
    deserialize=lambda buffer: pandas.core.indexes.datetimes.DatetimeIndex(
        string_deserialize(buffer)
    ),
)

recursive_serde_register(pandas.core.groupby.generic.DataFrameGroupBy)

recursive_serde_register(pandas._libs.lib._NoDefault)

recursive_serde_register(pandas.core.groupby.ops.BaseGrouper)

recursive_serde_register(pandas.core.groupby.grouper.Grouping)
recursive_serde_register(Subplot, exclude_attrs=["axes", "_axes", "_callbacks"])

recursive_serde_register(
    np.median, serialize=lambda x: "median", deserialize=lambda buffer: np.median
)

# recursive_serde_register(
#     np.mean,
#     serialize= lambda x: "mean",
#     deserialize= lambda buffer: np.mean)


# dont remove, we will need this when we introduce serialized plots
# recursive_serde_register(matplotlib.text.Text, exclude_attrs=["_axes",
#  "_agg_filter", "_callbacks", "_fontproperties", "_renderer"])
# recursive_serde_register(Subplot, exclude_attrs=["axes", "_axes",
#  "_callbacks", "_children", "_get_lines", "_get_patches_for_fill"])

# how else do you import a relative file to execute it?
NOTHING = None
