# stdlib
from enum import Enum
import json
import os
import subprocess
import sys

# third party
import numpy as np
import pandas as pd

# syft absolute
from syft.core.adp.data_subject_list import DataSubjectList  # noqa: F401


def auto_detect_domain_host_ip() -> str:
    ip_address = subprocess.check_output("echo $(curl -s ifconfig.co)", shell=True)
    domain_host_ip = ip_address.decode("utf-8").strip()
    if "google.colab" not in sys.modules:
        print(f"Your DOMAIN_HOST_IP is: {domain_host_ip}")
    else:
        print(
            "Google Colab detected, please manually set the `DOMAIN_HOST_IP` variable"
        )
        domain_host_ip = ""
    return domain_host_ip


class DatasetName(Enum):
    MEDNIST = "MedNIST"
    TISSUEMNIST = "TissueMNIST"


# Dataset Helper Methods
def get_label_mapping(file_name):
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
    else:
        raise ValueError(f"Not a valid Dataset : {file_name}")


def split_into_train_test_val_sets(data, test=0.20, val=0.10):
    train = 1.0 - (test + val)
    data.reset_index(inplace=True, drop=True)
    train_msk = np.random.rand(len(data)) < train
    train = data[train_msk]

    test_val = data[~train_msk]
    _val = (val * len(data)) / len(test_val)
    val_msk = np.random.rand(len(test_val)) < _val
    val = test_val[val_msk]
    test = test_val[~val_msk]

    # reset index
    train.reset_index(inplace=True, drop=True)
    val.reset_index(inplace=True, drop=True)
    test.reset_index(inplace=True, drop=True)

    return train, val, test


def load_data_as_df(file_path):
    df = pd.read_pickle(file_path)
    df.sort_values("patient_ids", inplace=True, ignore_index=True)

    # Get label mapping
    mapping = get_label_mapping(file_path)

    total_num = df.shape[0]
    print("Columns:", df.columns)
    print("Total Images:", total_num)
    print("Label Mapping", mapping)
    return df


def preprocess_data(data):

    # Convert images to numpy int64 array
    images = data["images"]
    images = np.dstack(images.values).astype(np.int64)  # type cast to int64
    dims = images.shape
    images = images.reshape(dims[0] * dims[1], dims[2])  # reshape to 2D array
    images = np.rollaxis(images, -1)

    # Convert labels to numpy int64 array
    labels = data["labels"].to_numpy().astype("int64")

    patient_ids = data["patient_ids"]

    return {"images": images, "labels": labels, "patient_ids": patient_ids}


def split_and_preprocess_dataset(data):

    print("Splitting dataset into train, validation and test sets.")
    train, val, test = split_into_train_test_val_sets(data)

    print("Preprocessing the dataset...")
    train_data = preprocess_data(train)
    val_data = preprocess_data(val)
    test_data = preprocess_data(test)
    print("Preprocessing completed.")
    return train_data, val_data, test_data


def get_data_description(data):
    unique_label_cnt = data.labels.nunique()
    lable_mapping = json.dumps(get_label_mapping())
    image_size = data.iloc[0]["images"].shape
    description = "The MedNIST dataset was gathered from several sets from TCIA, "
    description += "the RSNA Bone Age Challenge, and the NIH Chest X-ray dataset. "
    description += (
        "The dataset is kindly made available by Dr. Bradley J. Erickson M.D., Ph.D. "
    )
    description += "(Department of Radiology, Mayo Clinic) under the Creative Commons CC BY-SA 4.0 license.\n"
    description += f"Label Count: {unique_label_cnt}\n"
    description += f"Label Mapping: {lable_mapping}\n"
    description += f"Image Dimensions: {image_size}\n"
    description += f"Total Images: {data.shape[0]}\n"
    return description


def get_data_filename(dataset_url):
    return dataset_url.split("/")[-1]


def get_dataset_name(dataset_url):
    filename = dataset_url.split("/")[-1]
    return filename.split(".pkl")[0]


def download_dataset(dataset_url):
    filename = get_data_filename(dataset_url)
    if not os.path.exists(f"./{filename}"):
        os.system(f'curl -O "{dataset_url}"')
        print(f"{filename} is successfully downloaded.")
    else:
        print(f"{filename} is already downloaded")
    data = load_data_as_df(filename)
    return data


def validate_ds_credentials(ds_credentials):

    valid = True
    for key, val in ds_credentials.items():
        if not val:
            print(f"Please set a value for '{key}'.")
            valid = False
        elif key != "budget" and type(val) != str:
            print(f"Value for {key} needs to be a string.")
            valid = False

    if not valid:
        print("Please set the missing/incorrect values and re-run this cell")
    else:
        print("Data Scientist credentials are valid. Move to the next step.")
