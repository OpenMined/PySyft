# third party
import numpy as np
import pandas
import pandas as pd

# syft absolute
import syft as sy
from syft.core.adp.data_subject_list import DataSubjectList
from syft.core.tensor.autodp.ndim_entity_phi import NDimEntityPhiTensor as NDEPT


def make_ndept(data_file) -> NDEPT:
    df = pd.read_parquet(data_file)
    # name = f"Tweets- {df.shape[0]} rows dataset "
    impressions = ((np.array(list(df["impressions"])))).astype(np.int32)
    publication_title = list(df["publication_title"])
    entities = DataSubjectList.from_objs(["Tom"] * len(publication_title))

    ndept_data = sy.Tensor(impressions).private(
        min_val=0, max_val=30, entities=entities, ndept=True
    )

    return ndept_data
