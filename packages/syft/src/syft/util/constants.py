# third party
import numpy as np
import pandas as pd

DEFAULT_TIMEOUT = 5  # in seconds
SUPPORTED_RETURN_TYPES = (
    pd.DataFrame,
    pd.Series,
    np.ndarray,
    int,
    float,
    str,
    bytes,
    bool,
    tuple,
    dict,
    list,
    set,
    type(None),
)
