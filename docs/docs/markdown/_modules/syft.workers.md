# syft.workers package

## Submodules

## syft.workers.abstract module


#### class syft.workers.abstract.AbstractWorker()
Bases: `abc.ABC`


#### class syft.workers.abstract.IdProvider(given_ids=[])
Bases: `object`

Generate id and store the list of ids generated
Can take a pre set list in input and will complete
when it’s empty.


#### pop(\*args)
Provide random ids and store them

The syntax .pop() mimics the list syntax for convenience
and not the generator syntax

## syft.workers.base module


#### class syft.workers.base.BaseWorker(hook: syft.frameworks.torch.hook.TorchHook, id: Union[int, str] = 0, data: Union[List, tuple] = None, is_client_worker: bool = False, log_msgs: bool = False, verbose: bool = False)
Bases: `syft.workers.abstract.AbstractWorker`

Contains functionality to all workers.

Other workers will extend this class to inherit all functionality necessary
for PySyft’s protocol. Extensions of this class overrides two key methods
_send_msg() and _recv_msg() which are responsible for defining the
procedure for sending a binary message to another worker.

At it’s core, BaseWorker (and all workers) is a collection of objects owned
by a certain machine. Each worker defines how it interacts with objects on
other workers as well as how other workers interact with objects owned by
itself. Objects are either tensors or of any type supported by the PySyft
protocol.


* **Parameters**

    * **hook** – A reference to the TorchHook object which is used
      to modify PyTorch with PySyft’s functionality.

    * **id** – An optional string or integer unique id of the worker.

    * **known_workers** – An optional dictionary of all known workers on a
      network which this worker may need to communicate with in the
      future. The key of each should be each worker’s unique ID and
      the value should be a worker class which extends BaseWorker.
      Extensions of BaseWorker will include advanced functionality
      for adding to this dictionary(node discovery). In some cases,
      one can initialize this with known workers to help bootstrap
      the network.

    * **data** – Initialize workers with data on creating worker object

    * **is_client_worker** – An optional boolean parameter to indicate
      whether this worker is associated with an end user client. If
      so, it assumes that the client will maintain control over when
      variables are instantiated or deleted as opposed to handling
      tensor/variable/model lifecycle internally. Set to True if this
      object is not where the objects will be stored, but is instead
      a pointer to a worker that eists elsewhere.

    * **log_msgs** – An optional boolean parameter to indicate whether all
      messages should be saved into a log for later review. This is
      primarily a development/testing feature.



#### add_worker(worker: syft.workers.base.BaseWorker)
Adds a single worker.

Adds a worker to the list of _known_workers internal to the BaseWorker.
Endows this class with the ability to communicate with the remote
worker  being added, such as sending and receiving objects, commands,
or  information about the network.


* **Parameters**

    **worker** (`BaseWorker`) – A BaseWorker object representing the
    pointer to a remote worker, which must have a unique id.


### Example

```python
>>> import torch
>>> import syft as sy
>>> hook = sy.TorchHook(verbose=False)
>>> me = hook.local_worker
>>> bob = sy.VirtualWorker(id="bob",hook=hook, is_client_worker=False)
>>> me.add_worker([bob])
>>> x = torch.Tensor([1,2,3,4,5])
>>> x
1
2
3
4
5
[syft.core.frameworks.torch.tensor.FloatTensor of size 5]
>>> x.send(bob)
FloatTensor[_PointerTensor - id:9121428371 owner:0 loc:bob
            id@loc:47416674672]
>>> x.get()
1
2
3
4
5
[syft.core.frameworks.torch.tensor.FloatTensor of size 5]
```


#### add_workers(workers: List[BaseWorker])
Adds several workers in a single call.


* **Parameters**

    **workers** – A list of BaseWorker representing the workers to add.



#### clear_objects()
Removes all objects from the worker.


#### de_register_obj(obj: object, _recurse_torch_objs: bool = True)
Deregisters the specified object.

Deregister and remove attributes which are indicative of registration.


* **Parameters**

    * **obj** – A torch Tensor or Variable object to be deregistered.

    * **_recurse_torch_objs** – A boolean indicating whether the object is
      more complex and needs to be explored. Is not supported at the
      moment.



#### deserialized_search(query_items: Tuple[str])
Called when a message requesting a call to search is received.
The serialized arguments will arrive as a tuple and it needs to be
transformed to an arguments list.


* **Parameters**

    * **query_items** (*tuple**(**str**)*) – Tuple of items to search for. Should originate from the

    * **of a message requesting a search operation.** (*deserialization*) –



* **Returns**

    List of matched tensors.



* **Return type**

    list(PointerTensor)



#### execute_command(message: tuple)
Executes commands received from other workers.


* **Parameters**

    **message** – A tuple specifying the command and the args.



* **Returns**

    A pointer to the result.



#### fetch_plan(plan_id: Union[str, int])
Fetchs a copy of a the plan with the given plan_id from the worker registry.


* **Parameters**

    **plan_id** – A string indicating the plan id.



* **Returns**

    A plan if a plan with the given plan_id exists. Returns None otherwise.



#### generate_triple(cmd: Callable, field: int, a_size: tuple, b_size: tuple, locations: list)
Generates a multiplication triple and sends it to all locations.


* **Parameters**

    * **cmd** – An equation in einsum notation.

    * **field** – An integer representing the field size.

    * **a_size** – A tuple which is the size that a should be.

    * **b_size** – A tuple which is the size that b should be.

    * **locations** – A list of workers where the triple should be shared between.



* **Returns**

    A triple of AdditiveSharedTensors such that c_shared = cmd(a_shared, b_shared).



#### get_obj(obj_id: Union[str, int])
Returns the object from registry.

Look up an object from the registry using its ID.


* **Parameters**

    **obj_id** – A string or integer id of an object to look up.



#### static get_tensor_shape(tensor: torch.Tensor)
Returns the shape of a tensor casted into a list, to bypass the serialization of
a torch.Size object.


* **Parameters**

    **tensor** – A torch.Tensor.



* **Returns**

    A list containing the tensor shape.



#### get_worker(id_or_worker: Union[str, int, BaseWorker], fail_hard: bool = False)
Returns the worker id or instance.

Allows for resolution of worker ids to workers to happen automatically
while also making the current worker aware of new ones when discovered
through other processes.

If you pass in an ID, it will try to find the worker object reference
within self._known_workers. If you instead pass in a reference, it will
save that as a known_worker if it does not exist as one.

This method is useful because often tensors have to store only the ID
to a foreign worker which may or may not be known by the worker that is
de-serializing it at the time of deserialization.


* **Parameters**

    * **id_or_worker** – A string or integer id of the object to be returned
      or the BaseWorker object itself.

    * **fail_hard** (*bool*) – A boolean parameter indicating whether we want to
      throw an exception when a worker is not registered at this
      worker or we just want to log it.



* **Returns**

    A string or integer id of the worker or the BaseWorker instance
    representing the worker.


### Example

```python
>>> import syft as sy
>>> hook = sy.TorchHook(verbose=False)
>>> me = hook.local_worker
>>> bob = sy.VirtualWorker(id="bob",hook=hook, is_client_worker=False)
>>> me.add_worker([bob])
>>> bob
<syft.core.workers.virtual.VirtualWorker id:bob>
>>> # we can get the worker using it's id (1)
>>> me.get_worker('bob')
<syft.core.workers.virtual.VirtualWorker id:bob>
>>> # or we can get the worker by passing in the worker
>>> me.get_worker(bob)
<syft.core.workers.virtual.VirtualWorker id:bob>
```


#### static is_tensor_none(obj)

#### load_data(data: List[Union[torch.Tensor, syft.frameworks.torch.tensors.interpreters.abstract.AbstractTensor]])
Allows workers to be initialized with data when created

> The method registers the tensor individual tensor objects.


* **Parameters**

    **data** – A list of tensors



#### recv_msg(bin_message: bin)
Implements the logic to receive messages.

The binary message is deserialized and routed to the appropriate
function. And, the response serialized the returned back.

Every message uses this method.


* **Parameters**

    **bin_message** – A binary serialized message.



* **Returns**

    A binary message response.



#### register_obj(obj: object, obj_id: Union[str, int] = None)
Registers the specified object with the current worker node.

Selects an id for the object, assigns a list of owners, and establishes
whether it’s a pointer or not. This method is generally not used by the
client and is instead used by internal processes (hooks and workers).


* **Parameters**

    * **obj** – A torch Tensor or Variable object to be registered.

    * **obj_id** (*int** or **string*) – random integer between 0 and 1e10 or

    * **uniquely identifying the object.** (*string*) –



#### request_is_remote_tensor_none(pointer: syft.frameworks.torch.tensors.interpreters.pointer.PointerTensor)
Sends a request to the remote worker that holds the target a pointer if
the value of the remote tensor is None or not.
Note that the pointer must be valid: if there is no target (which is
different from having a target equal to None), it will return an error.


* **Parameters**

    **pointer** – The pointer on which we can to get information.



* **Returns**

    A boolean stating if the remote value is None.



#### request_obj(obj_id: Union[str, int], location: syft.workers.base.BaseWorker)
Returns the requested object from specified location.


* **Parameters**

    * **obj_id** – A string or integer id of an object to look up.

    * **location** – A BaseWorker instance that lets you provide the lookup
      location.



* **Returns**

    A torch Tensor or Variable object.



#### request_remote_tensor_shape(pointer: syft.frameworks.torch.tensors.interpreters.pointer.PointerTensor)
Sends a request to the remote worker that holds the target a pointer to
have its shape.


* **Parameters**

    **pointer** – A pointer on which we want to get the shape.



* **Returns**

    A torch.Size object for the shape.



#### respond_to_obj_req(obj_id: Union[str, int])
Returns the deregistered object from registry.


* **Parameters**

    **obj_id** – A string or integer id of an object to look up.



#### rm_obj(remote_key: Union[str, int])
Removes an object.

Remove the object from the permanent object registry if it exists.


* **Parameters**

    **remote_key** – A string or integer representing id of the object to be
    removed.



#### search(\*query)
Search for a match between the query terms and a tensor’s Id, Tag, or Description.

Note that the query is an AND query meaning that every item in the list of strings (query\*)
must be found somewhere on the tensor in order for it to be included in the results.


* **Parameters**

    * **query** – A list of strings to match against.

    * **me** – A reference to the worker calling the search.



* **Returns**

    A list of PointerTensors.



#### send(obj: Union[torch.Tensor, syft.frameworks.torch.tensors.interpreters.abstract.AbstractTensor], workers: syft.workers.base.BaseWorker, ptr_id: Union[str, int] = None)
Sends tensor to the worker(s).

Send a syft or torch tensor/object and its child, sub-child, etc (all the
syft chain of children) to a worker, or a list of workers, with a given
remote storage address.


* **Parameters**

    * **tensor** – A syft/torch tensor/object object to send.

    * **workers** – A BaseWorker object representing the worker(s) that will
      receive the object.

    * **ptr_id** – An optional string or integer indicating the remote id of
      the object on the remote worker(s).


### Example

```python
>>> import torch
>>> import syft as sy
>>> hook = sy.TorchHook(torch)
>>> bob = sy.VirtualWorker(hook)
>>> x = torch.Tensor([1, 2, 3, 4])
>>> x.send(bob, 1000)
Will result in bob having the tensor x with id 1000
```


* **Returns**

    A PointerTensor object representing the pointer to the remote worker(s).



#### send_command(recipient: syft.workers.base.BaseWorker, message: str, return_ids: str = None)
Sends a command through a message to a recipient worker.


* **Parameters**

    * **recipient** – A recipient worker.

    * **message** – A string representing the message being sent.

    * **return_ids** – A list of strings indicating the ids of the
      tensors that should be returned as response to the command execution.



* **Returns**

    A list of PointerTensors or a single PointerTensor if just one response is expected.



#### send_msg(msg_type: int, message: str, location: syft.workers.base.BaseWorker)
Implements the logic to send messages.

The message is serialized and sent to the specified location. The
response from the location (remote worker) is deserialized and
returned back.

Every message uses this method.


* **Parameters**

    * **msg_type** – A integer representing the message type.

    * **message** – A string representing the message being received.

    * **location** – A BaseWorker instance that lets you provide the
      destination to send the message.



* **Returns**

    The deserialized form of message from the worker at specified
    location.



#### send_obj(obj: object, location: syft.workers.base.BaseWorker)
Send a torch object to a worker.


* **Parameters**

    * **obj** – A torch Tensor or Variable object to be sent.

    * **location** – A BaseWorker instance indicating the worker which should
      receive the object.



#### set_obj(obj: Union[torch.Tensor, syft.frameworks.torch.tensors.interpreters.abstract.AbstractTensor])
Adds an object to the registry of objects.


* **Parameters**

    **obj** – A torch or syft tensor with an id


## syft.workers.plan module


#### class syft.workers.plan.Plan(hook, owner, name='', \*args, \*\*kwargs)
Bases: `syft.workers.base.BaseWorker`

This worker does not send messages or execute any commands. Instead,
it simply records messages that are sent to it such that message batches
(called ‘Plans’) can be created and sent once.


#### build_plan(args)
The plan must be built with some input data, here args. When they
are provided, they are sent to the plan worker, which executes its
blueprint: each command of the blueprint is catched by _recv_msg
and is used to fill the plan
:param args: the input data


#### copy()

#### describe(description)

#### execute_plan(args, result_ids)
Control local or remote plan execution.
If the plan doesn’t have the plan built, first build it using the blueprint.
Then if it has a remote location, send the plan to the remote location only the
first time, request a remote plan execution with specific pointers and ids for
storing the result, and return a pointer to the result of the execution.
If the plan is local: update the plan with the result_ids and args ids given,
run the plan and return the None message serialized.


#### get()
Mock get function: no call to remote worker is made, we just erase the information
linking this plan to that remote worker.


#### replace_ids(from_ids, to_ids)
Replace pairs of tensor ids in the plan stored
:param from_ids: the left part of the pair: ids to change
:param to_ids: the right part of the pair: ids to replace with


#### replace_worker_ids(from_worker_id, to_worker_id)
Replace occurrences of from_worker_id by to_worker_id in the plan stored
Works also if those ids are encoded in bytes (for string)


#### request_execute_plan(response_ids, \*args, \*\*kwargs)
Send a request to execute the plan on the remote location
:param response_ids: where the plan result should be stored remotely
:param args: the arguments use as input data for the plan
:return:


#### send(location)
Mock send function that only specify that the Plan will have to be sent to location.
In a way, when one calls .send(), this doesn’t trigger a call to a remote worker, but
just stores “a promise” that it will be sent (with _send()) later when the plan in
called (and built)


#### tag(\*_tags)

#### syft.workers.plan.func2plan(plan_blueprint)
the @func2plan decorator - converts a function of pytorch code into a plan object
which can be sent to any arbitrary worker.


#### syft.workers.plan.make_plan(plan_blueprint)
For folks who would prefer to not use a decorator, they can use this function


#### syft.workers.plan.method2plan(plan_blueprint)
the @method2plan decorator - converts a method containing sequential pytorch code into
a plan object which can be sent to any arbitrary worker.

## syft.workers.virtual module


#### class syft.workers.virtual.VirtualWorker(hook: syft.frameworks.torch.hook.TorchHook, id: Union[int, str] = 0, data: Union[List, tuple] = None, is_client_worker: bool = False, log_msgs: bool = False, verbose: bool = False)
Bases: `syft.workers.base.BaseWorker`

## syft.workers.websocket_client module


#### class syft.workers.websocket_client.WebsocketClientWorker(hook, host: str, port: int, id: Union[int, str] = 0, is_client_worker: bool = False, log_msgs: bool = False, verbose: bool = False, data: List[Union[torch.Tensor, syft.frameworks.torch.tensors.interpreters.abstract.AbstractTensor]] = None)
Bases: `syft.workers.base.BaseWorker`


#### search(\*query)
Search for a match between the query terms and a tensor’s Id, Tag, or Description.

Note that the query is an AND query meaning that every item in the list of strings (query\*)
must be found somewhere on the tensor in order for it to be included in the results.


* **Parameters**

    * **query** – A list of strings to match against.

    * **me** – A reference to the worker calling the search.



* **Returns**

    A list of PointerTensors.


## syft.workers.websocket_server module


#### class syft.workers.websocket_server.WebsocketServerWorker(hook, host: str, port: int, id: Union[int, str] = 0, log_msgs: bool = False, verbose: bool = False, data: List[Union[torch.Tensor, syft.frameworks.torch.tensors.interpreters.abstract.AbstractTensor]] = None, loop=None)
Bases: `syft.workers.virtual.VirtualWorker`


#### start()
Start the server

## Module contents


#### syft.workers.func2plan(plan_blueprint)
the @func2plan decorator - converts a function of pytorch code into a plan object
which can be sent to any arbitrary worker.


#### syft.workers.method2plan(plan_blueprint)
the @method2plan decorator - converts a method containing sequential pytorch code into
a plan object which can be sent to any arbitrary worker.


#### syft.workers.make_plan(plan_blueprint)
For folks who would prefer to not use a decorator, they can use this function


#### class syft.workers.IdProvider(given_ids=[])
Bases: `object`

Generate id and store the list of ids generated
Can take a pre set list in input and will complete
when it’s empty.


#### pop(\*args)
Provide random ids and store them

The syntax .pop() mimics the list syntax for convenience
and not the generator syntax
