# stdlib
import functools

# third party
import numpy as np
import pandas as pd
import pyperf

# syft absolute
import syft as sy
from syft.core.adp.data_subject import DataSubject


def create_bench_constructor(
    runner: pyperf.Runner,
    data_file,
) -> None:

    df = pd.read_parquet(data_file)
    # name = f"Tweets- {df.shape[0]} rows dataset "
    impressions = np.array(list(df["impressions"]))
    publication_title = list(df["publication_title"])
    data_subjects = list()
    for i in range(len(publication_title)):
        data_subjects.append(DataSubject(name=publication_title[i]))

    partial_function_evaluation = functools.partial(
        sy.Tensor(impressions).private,
        min_val=0,
        max_val=30,
        data_subjects=data_subjects,
    )
    rows = len(impressions)
    row_type = "phitensor"
    runner.bench_func(
        f"constructor_for_{row_type}_rows_{rows:,}",
        partial_function_evaluation,
    )
