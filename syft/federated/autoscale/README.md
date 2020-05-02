# Autoscale on GCP

To use this feature please install:

- [Terraform](https://learn.hashicorp.com/terraform/getting-started/install.html)

- [Terrascript](https://github.com/mjuenema/python-terrascript)

```bash
pip install terrascript
```

## Usage

You can create compute instance using the sample code in test .py

- Initialize using :

```bash

instance_name = gcloud.GoogleCloud(
    credentials="GCP Login/terraf.json",
    project_id="terraform",
    region="us-central1"
)
```

- Create Instnaces using :

```bash
instance_name.compute_instance(
    name="new-12345",
    machine_type="f1-micro",
    zone="us-central1-a",
    image_family="debian-9",
)
```

- Destroy the created instances using :

```bash
instance_name.destroy()
