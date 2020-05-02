# Autoscale on GCP

## Setting up GCP

In addition to a GCP account, you'll need two things to use Terraform to provision your infrastructure:

- A GCP Project:
 GCP organizes resources into projects. Create one now in the GCP console. You'll need the Project ID later. You can see a list of your projects in the cloud resource manager.

- Google Compute Engine: You'll need to enable Google Compute Engine for your project. Do so now in the console. Make sure the project you're using to follow this guide is selected and click the "Enable" button.

- A GCP service account key: Terraform will access your GCP account by using a service account key. Create one now in the console. When creating the key, use the following settings:

  - Select the project you created in the previous step.
  - Under "Service account", select "New service account".
  - Give it any name you like.
  - For the Role, choose "Project -> Editor".
  - Leave the "Key Type" as JSON.
  - Click "Create" to create the key and save the key file to your system.
  - You can read more about service account keys in Google's documentation.

## Dependencies

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
