# syft.frameworks.torch.federated package

## Submodules

## syft.frameworks.torch.federated.dataloader module


#### class syft.frameworks.torch.federated.dataloader.FederatedDataLoader(federated_dataset, batch_size=8, shuffle=False, num_iterators=1, drop_last=False, collate_fn=<function default_collate>, iter_per_worker=False, \*\*kwargs)
Bases: `object`

Data loader. Combines a dataset and a sampler, and provides
single or several iterators over the dataset.


* **Parameters**

    * **federated_dataset** (*FederatedDataset*) – dataset from which to load the data.

    * **batch_size** (*int**, **optional*) – how many samples per batch to load
      (default: `1`).

    * **shuffle** (*bool**, **optional*) – set to `True` to have the data reshuffled
      at every epoch (default: `False`).

    * **collate_fn** (*callable**, **optional*) – merges a list of samples to form a mini-batch.

    * **drop_last** (*bool**, **optional*) – set to `True` to drop the last incomplete batch,
      if the dataset size is not divisible by the batch size. If `False` and
      the size of dataset is not divisible by the batch size, then the last batch
      will be smaller. (default: `False`)

    * **num_iterators** (*int*) – number of workers from which to retrieve data in parallel.
      num_iterators <= len(federated_dataset.workers) - 1
      the effect is to retrieve num_iterators epochs of data but at each step data from num_iterators distinct
      workers is returned.

    * **iter_per_worker** (*bool*) – if set to true, __next__() will return a dictionary containing one batch per worker



#### syft.frameworks.torch.federated.dataloader.default_collate(batch)
Puts each data field into a tensor with outer dimension batch size

## syft.frameworks.torch.federated.dataset module


#### class syft.frameworks.torch.federated.dataset.BaseDataset(data, targets)
Bases: `object`

This is a base class to used for manipulating a dataset. This is composed
of a .data attribute for inputs and a .targets one for labels. It is to
be used like the MNIST Dataset object, and is useful to avoid handling
the two inputs and label tensors separately.


#### federate(workers)
Add a method to easily transform a torch.Dataset or a sy.BaseDataset
into a sy.FederatedDataset. The dataset given is split in len(workers)
part and sent to each workers


#### fix_prec(\*args, \*\*kwargs)

#### fix_precision(\*args, \*\*kwargs)

#### float_prec(\*args, \*\*kwargs)

#### float_precision(\*args, \*\*kwargs)

#### get()

#### location()

#### send(worker)

#### share(\*args, \*\*kwargs)

#### class syft.frameworks.torch.federated.dataset.FederatedDataset(datasets)
Bases: `object`


#### workers()

#### syft.frameworks.torch.federated.dataset.dataset_federate(dataset, workers)
Add a method to easily transform a torch.Dataset or a sy.BaseDataset
into a sy.FederatedDataset. The dataset given is split in len(workers)
part and sent to each workers

## syft.frameworks.torch.federated.utils module


#### syft.frameworks.torch.federated.utils.add_model(dst_model, src_model)
Add the parameters of two models.


* **Parameters**

    * **dst_model** (*torch.nn.Module*) – the model to which the src_model will be added

    * **src_model** (*torch.nn.Module*) – the model to be added to dst_model



* **Returns**

    the resulting model of the addition



* **Return type**

    torch.nn.Module



#### syft.frameworks.torch.federated.utils.extract_batches_per_worker(federated_train_loader: syft.frameworks.torch.federated.dataloader.FederatedDataLoader)
Extracts the batches from the federated_train_loader and stores them
in a dictionary (keys = data.location)

Args:
federated_train_loader: the connection object we use to send responses

> back to the client.


#### syft.frameworks.torch.federated.utils.federated_avg(models: List[torch.nn.modules.module.Module])
Calculate the federated average of a list of models.


* **Parameters**

    **models** (*List**[**torch.nn.Module**]*) – the models of which the federated average is calculated



* **Returns**

    the module with averaged parameters



* **Return type**

    torch.nn.Module



#### syft.frameworks.torch.federated.utils.scale_model(model, scale)
Scale the parameters of a model.


* **Parameters**

    * **model** (*torch.nn.Module*) – the models whose parameters will be scaled

    * **scale** (*float*) – the scaling factor



* **Returns**

    the module with scaled parameters



* **Return type**

    torch.nn.Module


## Module contents
