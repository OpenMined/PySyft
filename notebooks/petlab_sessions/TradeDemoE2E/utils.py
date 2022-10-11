# stdlib
from enum import Enum
import json
import os
import subprocess
import sys
import uuid

# third party
from IPython import get_ipython
from IPython.display import Javascript  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

# syft absolute
from syft.core.adp.data_subject_list import DataSubjectList  # noqa: F401


def auto_detect_domain_host_ip(silent: bool = False) -> str:
    ip_address = subprocess.check_output("echo $(curl -s ifconfig.co)", shell=True)
    domain_host_ip = ip_address.decode("utf-8").strip()
    if "google.colab" not in sys.modules:
        if not silent:
            print(f"Your DOMAIN_HOST_IP is: {domain_host_ip}")
    else:
        if not silent:
            print(
                "Google Colab detected, please manually set the `DOMAIN_HOST_IP` variable"
            )
        domain_host_ip = ""
    return domain_host_ip


def download_dataset_as_dataframe(dataset_url):
    def load_data_as_df(file_path):
        df = pd.read_csv(file_path)
        unique_trade_flows = df["Trade Flow"].unique()
        print("Data shape:", df.shape, end="\n\n")
        print("Columns:", df.columns, end="\n\n")
        print("Unique Trade Flows:", unique_trade_flows, end="\n\n")

        return df

    data = load_data_as_df(dataset_url)

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


def output_dataset_url():
    return """
    var a = window.location.href.split('#')
    if (a.length > 1) {
        element.textContent = 'MY_DATASET_URL="https://raw.githubusercontent.com/OpenMined/datasets/main/TissueMNIST/subsets/TissueMNIST-' + a[1] + '.pkl"'
    } else {
        element.textContent = 'Unable to automatically get MY_DATASET_URL please locate it from your session details.'
    }
    """  # noqa: E501
