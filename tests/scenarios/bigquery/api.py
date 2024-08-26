# third party
from unsync import unsync


@unsync
def get_datasets(client):
    print("Checking datasets")
    num_datasets = len(client.api.services.dataset.get_all())
    print(">>> num datasets", num_datasets)
    return num_datasets
