# Dataset flow

**Creating a dataset (DO)**:

```python
do_manager.create_dataset(
    name="my dataset",
    mock_path="/path/to/mock/data",      # Public mock data
    private_path="/path/to/private/data", # Private data (never leaves DO)
    summary="Dataset description",
    users=[ds_email],  # Who can access
)
```

**Reading a dataset (DS)**:

```python
ds_manager.sync()  # Pull available datasets
datasets = ds_manager.datasets.get_all()
dataset = ds_manager.datasets.get("my dataset", datasite=do_email)
# Only mock_files are available to DS, private_files stay with DO
```
