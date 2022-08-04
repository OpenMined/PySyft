#!/usr/bin/env python
# coding: utf-8
# stdlib
import sys

# third party
import numpy as np

# syft absolute
import syft as sy
from syft.core.adp.data_subject_list import DataSubjectArray

# relative
from .utils import download_dataset
from .utils import split_and_preprocess_dataset


def add_dataset_to_domain(domain_ip, dataset_url):

    # Need to run update_syft.sh on every domain
    #         sshpass -p MflpTeamOne@31052022 ssh azureuser@20.253.155.189 update_syft.sh

    # Log into domain
    domain = sy.login(email="info@openmined.org", password="changethis", url=domain_ip)

    # Preprocess dataset split
    dataset = download_dataset(dataset_url)
    train, val, test = split_and_preprocess_dataset(data=dataset)

    data_subjects_image = np.ones(train["images"].shape).astype(object)
    for i, patient in enumerate(train["patient_ids"]):
        data_subjects_image[i] = DataSubjectArray([str(patient)])

    data_subjects_labels = np.ones(train["labels"].shape).astype(object)
    for i, patient in enumerate(train["patient_ids"]):
        data_subjects_labels[i] = DataSubjectArray([str(patient)])

    train_image_data = sy.Tensor(train["images"]).annotated_with_dp_metadata(
        min_val=0, max_val=255, data_subjects=data_subjects_image
    )
    train_label_data = sy.Tensor(train["labels"]).annotated_with_dp_metadata(
        min_val=0, max_val=1, data_subjects=data_subjects_labels
    )

    # Load dataset
    domain.load_dataset(
        name="BreastCancerDataset",
        assets={
            "train_images": train_image_data[:5],
            "train_labels": train_label_data[:5],
        },
        description="Invasive Ductal Carcinoma (IDC) is the most common subtype of all breast cancers. \
            The modified dataset consisted of 162 whole mount slide images of Breast Cancer (BCa) specimens \
            scanned at 40x.Patches of size 50 x 50 were extracted from the original image. \
            The labels 0 is non-IDC and 1 is IDC.",
    )
    try:
        domain.create_user(
            name="Sam Carter",
            email="sam@stargate.net",
            password="changethis",
            budget=9999,
        )
    except Exception as e:
        print(e)


if __name__ == "__main__":
    domain_ip, dataset_url = sys.argv[1], sys.argv[2]
    add_dataset_to_domain(domain_ip, dataset_url.strip("\r"))
    print("Finished Uploading dataset")
