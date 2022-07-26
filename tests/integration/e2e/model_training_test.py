# stdlib
import time

# third party
import numpy as np

# syft absolute
import syft as sy
from syft import nn
from syft.core.adp.data_subject_list import DataSubjectArray
from syft.core.adp.utils import download_dataset
from syft.core.adp.utils import split_and_preprocess_dataset

DOMAIN1_PORT = 9082
DS_EMAIL = "sam@stargate.net"
DS_PASSWORD = "changethis"
DATASET_URL = (
    "https://raw.githubusercontent.com/OpenMined/datasets/main"
    + "/BreastCancerDataset/subsets/BreastCancerDataset-02ec48b840824b1ea3e1f5d11c45314b.pkl"
)


def add_dataset_to_domain():
    dataset_url = DATASET_URL

    # Log into domain
    domain = sy.login(
        email="info@openmined.org", password="changethis", port=DOMAIN1_PORT
    )

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


def initialize_model(input_shape) -> nn.Model:

    # create an empty model
    model = nn.Model()

    # Add layers to the model

    # Layer 1
    model.add(
        nn.Convolution(nb_filter=32, filter_size=3, padding=2, input_shape=input_shape)
    )
    model.add(nn.BatchNorm(activation="leaky_relu"))
    model.add(nn.MaxPool(pool_size=2, stride=2))

    # Layer 2
    # model.add(nn.Convolution(nb_filter=64, filter_size=3, padding=2))
    # model.add(nn.BatchNorm(activation="leaky_relu"))
    # model.add(nn.MaxPool(pool_size=2, stride=2))

    # Layer 3
    model.add(nn.AvgPool(3))

    # Layer 4
    model.add(nn.Flatten())

    # Layer 5
    model.add(nn.Linear(2, 512))

    model.initialize_weights()

    return model


def test_model_training():
    """Remotely train on the list of domains."""

    add_dataset_to_domain()

    # Get input shape of the images
    # We assume images on all domains have the same shape.
    domain = sy.login(port=DOMAIN1_PORT, email=DS_EMAIL, password=DS_PASSWORD)
    X_train = domain.datasets[-1]["train_images"][:2]  # just taking first two images.
    input_shape = X_train.public_shape  # Shape Format -> (N, C, H, W)

    print("Input Shape: ", input_shape)

    # initialize the model
    model = initialize_model(input_shape=input_shape)

    # Log into the domain
    domain = sy.login(port=DOMAIN1_PORT, email=DS_EMAIL, password=DS_PASSWORD)

    # Check if dataset is present on the domain
    if len(domain.datasets) == 0:
        raise ValueError("Domain does not have any dataset for Model Training")

    # Get pointers to images and labels
    X_train = domain.datasets[-1]["train_images"][:2]
    y_train = domain.datasets[-1]["train_labels"][:2]

    print(f"Image Shape: {X_train.public_shape}, Label Shape: {y_train.public_shape}")

    # Perform a single epoch
    model_ptr = model.send(domain, send_to_blob_storage=False)

    # Perform a single step
    # A single step performs the following operation on the batch of images:
    # - forward pass
    # - calculates loss
    # - backward pass
    # - optimizer weight updates

    print("waiting for model ptr to be ready !!!")
    # wait for model_ptr to be accessible
    while not model_ptr.exists:
        time.sleep(2)

    run_status = model_ptr.step(X_train, y_train)

    # Wait for step operation to be completed on the remote domain.
    while not run_status.exists:
        time.sleep(10)

    # Publish Model Weights
    print(f"Publishing model weights on Domain ({domain.name})")
    published_obj = model_ptr.publish(
        sigma=1e4
    )  # PB spent with 1e3 = 690k, thus 1e4 sigma -> 6.9k which is within PB limit of 9999
    while not published_obj.exists:
        time.sleep(2)
    parameters = published_obj.get_copy()

    loss = parameters["loss"]
    print(f"Model loss on Domain  ({domain.name}): {loss}")

    # Update weights and move onto next domain node
    model = initialize_model(input_shape)
    model.replace_weights(parameters)
    print(f"Model training finished on Domain: - {domain.name}")

    print()
