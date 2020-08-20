# Autoscale on GCP

## Setting up GCP

In addition to a GCP account, you'll need two things to use Terraform to provision your infrastructure:

- A GCP Project:
 GCP organizes resources into projects. Create one now in the GCP console. You'll need the Project ID later. You can see a list of your projects in the cloud resource manager.

- Google Compute Engine: You'll need to enable Google Compute Engine for your project. Do so now in the console. Make sure the project you're using to follow this guide is selected and click the "Enable" button.

- A [GCP service account](https://cloud.google.com/iam/docs/creating-managing-service-accounts) key: Terraform will access your GCP account by using a service account key. Create one now in the console. When creating the key, use the following settings:

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

### Set Up Budget Alerts (important)

Before you start to spin-up instances we encourace to set a budget alert on GCP to avoid suprise costs.

[Setup Budget and Budget Alerts](https://cloud.google.com/billing/docs/how-to/budgets)

### Spin-up Instances using the follwoing commands:-

You can find sample code in ```test.py```  and  ```test.ipynb```

- Import enums from the ```gcloud_configurations.py```

```python
import syft.grid.autoscale.utils.gcloud_configurations as configs
```

- Initialize using:

```python

instance_name = gcloud.GoogleCloud(
    credentials="GCP Login/terraf.json",
    project_id="terraform",
    region=configs.Region.us_central1,
)
```

- Reserve IP address using:

```python
instance_name.reserve_ip("grid")
```

- Create Instances using:

```python
instance_name.compute_instance(
    name="new-12345",
    machine_type=configs.MachineType.f1_micro,
    zone=configs.Zone.us_central1_a,
    image_family=configs.ImageFamily.ubuntu_2004_lts,
)
```

- Create PyGrid Network instance using:

```python
instance_name.create_gridnetwork(
    name="new-network",
    machine_type=configs.MachineType.f1_micro,
    zone=configs.Zone.us_central1_a,
)
```

- Create PyGrid Node instance using:

```python
instance_name.create_gridnode(
    name="new-node",
    machine_type=configs.MachineType.f1_micro,
    zone=configs.Zone.us_central1_a,
    gridnetwork_name="new-network",
)
```

- Create Clusters using:

```python
c1 = instance_name.create_cluster(
    name="my-cluster1",
    machine_type=configs.MachineType.f1_micro,
    zone=configs.Zone.us_central1_a,
    reserve_ip_name="grid"
    target_size=3,
    eviction_policy="delete",
)
```

- Run a parameter sweep to figure out the best parameters using:

```python
c1.sweep()
```

- Destroy the created instances using:

```python
instance_name.destroy()
```
