# stdlib
from enum import Enum
import os
from typing import Any
from typing import Dict
from typing import Tuple

# third party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class DatasetName(Enum):
    MEDNIST = "MedNIST"
    TISSUEMNIST = "TissueMNIST"
    BREASTCANCERDATASET = "BreastCancerDataset"


# Dataset Helper Methods
def get_label_mapping(file_name: str) -> Dict[str, int]:
    # the data uses the following mapping
    if DatasetName.MEDNIST.value in file_name:
        return {
            "AbdomenCT": 0,
            "BreastMRI": 1,
            "CXR": 2,
            "ChestCT": 3,
            "Hand": 4,
            "HeadCT": 5,
        }
    elif DatasetName.TISSUEMNIST.value in file_name:
        return {
            "Collecting Duct, Connecting Tubule": 0,
            "Distal Convoluted Tubule": 1,
            "Glomerular endothelial cells": 2,
            "Interstitial endothelial cells": 3,
            "Leukocytes": 4,
            "Podocytes": 5,
            "Proximal Tubule Segments": 6,
            "Thick Ascending Limb": 7,
        }
    elif DatasetName.BREASTCANCERDATASET.value in file_name:
        return {
            "Non-Invasive Ductal Carcinoma (IDC)": 0,
            "Invasive Ductal Carcinoma (IDC)": 1,
        }
    else:
        raise ValueError(f"Not a valid Dataset : {file_name}")


def split_into_train_test_val_sets(
    data: pd.DataFrame, test: float = 0.10, val: float = 0.10
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = 1.0 - (test + val)
    data.reset_index(inplace=True, drop=True)
    train_msk = np.random.rand(len(data)) < train
    train_df = data[train_msk]

    test_val = data[~train_msk]
    _val = (val * len(data)) / len(test_val)
    val_msk = np.random.rand(len(test_val)) < _val
    val_df = test_val[val_msk]
    test_df = test_val[~val_msk]

    # reset index
    train_df.reset_index(inplace=True, drop=True)
    val_df.reset_index(inplace=True, drop=True)
    test_df.reset_index(inplace=True, drop=True)

    return train_df, val_df, test_df


def load_data_as_df(file_path: str) -> pd.DataFrame:
    df = pd.read_pickle(file_path)
    df.sort_values("patient_ids", inplace=True, ignore_index=True)

    # Get label mapping
    mapping = get_label_mapping(file_path)

    total_num = df.shape[0]
    print("Columns:", df.columns)
    print("Total Images:", total_num)
    print("Label Mapping", mapping)
    return df


def preprocess_data(data: pd.DataFrame) -> Dict[str, Any]:
    # TODO: Fix to consider all types of datasets
    # Convert images to numpy int64 array
    images = data["images"]
    reshaped_images = []
    for i in range(len(images)):
        img = images[i]
        if ((50, 50, 3)) != images[i].shape:
            img = np.resize(img, (50, 50, 3))
        dims = img.shape
        img = img.reshape(dims[2], dims[0], dims[1]).astype(np.int64)
        reshaped_images.append(img)

    #     images = np.vstack(reshaped_images).astype(np.int64)  # type cast to int64
    images = np.array(reshaped_images)
    dims = images.shape
    print("Dims", dims)
    # images = images.reshape(dims[0] * dims[1], dims[2])  # reshape to 2D array
    #     images = np.rollaxis(images, -1)

    # Convert labels to numpy int64 array
    labels = data["labels"].to_numpy().astype("int64")

    patient_ids = data["patient_ids"].values

    return {"images": images, "labels": labels, "patient_ids": patient_ids}


def split_and_preprocess_dataset(
    data: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("Splitting dataset into train, validation and test sets.")
    train, val, test = split_into_train_test_val_sets(data)

    print("Preprocessing the dataset...")
    train_data = preprocess_data(train)
    val_data = preprocess_data(val)
    test_data = preprocess_data(test)
    print("Preprocessing completed.")
    return train_data, val_data, test_data


def get_data_filename(dataset_url: str) -> str:
    return dataset_url.split("/")[-1]


def download_dataset(dataset_url: str) -> pd.DataFrame:
    filename = get_data_filename(dataset_url)
    if not os.path.exists(f"./{filename}"):
        os.system(f'curl -O "{dataset_url}"')
        print(f"{filename} is successfully downloaded.")
    else:
        print(f"{filename} is already downloaded")
    data = load_data_as_df(filename)
    fig, ax = plt.subplots(5, 10, figsize=(20, 10))

    fig.suptitle("\nBreast Histopathology Images", fontsize=24)
    selection = np.random.choice(data.index.values, size=50)

    for n in range(5):
        for m in range(10):
            idx = selection[m + 10 * n]
            image = data.loc[idx, "images"]
            ax[n, m].imshow(image)
            ax[n, m].grid(False)
    return data
