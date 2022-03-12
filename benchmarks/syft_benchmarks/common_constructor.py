# stdlib
import functools

# third party
import numpy as np
import pandas as pd
import pyperf

# syft absolute
import syft as sy
from syft.core.adp.entity import Entity


def create_bench_constructor(
    runner: pyperf.Runner,
    data_file,
    ndept: bool = False,
) -> None:

    df = pd.read_parquet(data_file)
    # name = f"Tweets- {df.shape[0]} rows dataset "
    impressions = ((np.array(list(df["impressions"])))).astype(np.int32)
    publication_title = list(df["publication_title"])
    entities = list()
    for i in range(len(publication_title)):
        entities.append(Entity(name=publication_title[i]))

    partial_function_evaluation = functools.partial(
        sy.Tensor(impressions).private,
        min_val=0,
        max_val=30,
        entities=entities,
        ndept=ndept,
    )
    rows = len(impressions)
    row_type = "ndept" if ndept else "rept"
    runner.bench_func(
        f"constructor_for_{row_type}_rows_{rows:,}",
        partial_function_evaluation,
    )
