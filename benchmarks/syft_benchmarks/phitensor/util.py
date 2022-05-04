# third party
import numpy as np
import pandas
import pandas as pd

# syft absolute
import syft as sy
from syft.core.adp.data_subject_list import DataSubjectList
from syft.core.tensor.autodp.phi_tensor import PhiTensor as PT


def make_phitensor(data_file) -> PT:
    df = pd.read_parquet(data_file)
    # name = f"Tweets- {df.shape[0]} rows dataset "
    impressions = np.array(list(df["impressions"]))
    publication_title = list(df["publication_title"])
    data_subjects = DataSubjectList.from_objs(["Tom"] * len(publication_title))

    phitensor_data = sy.Tensor(impressions).private(
        min_val=0, max_val=30, data_subjects=data_subjects
    )

    return phitensor_data
