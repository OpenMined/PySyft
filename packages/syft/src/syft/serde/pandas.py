# stdlib
import io
import json
from typing import Any

# third party
import numpy
import numpy as np
import pandas
from pandas.core import resample

# syft absolute
import syft as sy

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

recursive_serde_register(resample.DatetimeIndexResampler)

recursive_serde_register(resample.TimeGrouper)

recursive_serde_register(pandas.core.groupby.ops.BinGrouper)

recursive_serde_register(
    pandas._libs.tslibs.offsets.MonthEnd,
    serialize=lambda x: "np.MontEnd",
    deserialize=lambda buffer: pandas._libs.tslibs.offsets.MonthEnd(),
)

recursive_serde_register(
    np.median, serialize=lambda x: "np.median", deserialize=lambda buffer: np.median
)

recursive_serde_register(
    np.mean, serialize=lambda x: "np.mean", deserialize=lambda buffer: np.mean
)

recursive_serde_register(sum, serialize=lambda x: "sum", deserialize=lambda buffer: sum)

recursive_serde_register(
    pandas.core.strings.accessor.StringMethods,
    serialize=lambda x: sy.serialize(x._data, to_bytes=True),
    deserialize=lambda buffer: pandas.core.strings.accessor.StringMethods(
        sy.deserialize(buffer, from_bytes=True)
    ),
)

recursive_serde_register(
    pandas.core.indexing._LocIndexer,
    serialize=lambda x: sy.serialize(x.obj, to_bytes=True),
    deserialize=lambda buffer: sy.deserialize(buffer, from_bytes=True).loc,
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
# recursive_serde_register(Subplot, exclude_attrs=["axes", "_axes", "_callbacks"])

# how else do you import a relative file to execute it?
NOTHING = None
