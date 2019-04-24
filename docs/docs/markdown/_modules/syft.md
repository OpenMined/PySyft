# syft package

## Subpackages

* syft.frameworks package

  * Subpackages

    * syft.frameworks.torch package

      * Subpackages

        * syft.frameworks.torch.crypto package

          * Submodules

          * syft.frameworks.torch.crypto.securenn module

          * syft.frameworks.torch.crypto.spdz module

          * Module contents

        * syft.frameworks.torch.differential_privacy package

          * Submodules

          * syft.frameworks.torch.differential_privacy.pate module

          * Module contents

        * syft.frameworks.torch.federated package

          * Submodules

          * syft.frameworks.torch.federated.dataloader module

          * syft.frameworks.torch.federated.dataset module

          * syft.frameworks.torch.federated.utils module

          * Module contents

        * syft.frameworks.torch.tensors package

          * Subpackages

            * syft.frameworks.torch.tensors.decorators package

              * Submodules

              * syft.frameworks.torch.tensors.decorators.logging module

              * Module contents

            * syft.frameworks.torch.tensors.interpreters package

              * Submodules

              * syft.frameworks.torch.tensors.interpreters.Polynomial module

              * syft.frameworks.torch.tensors.interpreters.abstract module

              * syft.frameworks.torch.tensors.interpreters.additive_shared module

              * syft.frameworks.torch.tensors.interpreters.multi_pointer module

              * syft.frameworks.torch.tensors.interpreters.native module

              * syft.frameworks.torch.tensors.interpreters.plusisminus module

              * syft.frameworks.torch.tensors.interpreters.pointer module

              * syft.frameworks.torch.tensors.interpreters.precision module

              * Module contents

          * Module contents

      * Submodules

      * syft.frameworks.torch.functions module

      * syft.frameworks.torch.hook module

      * syft.frameworks.torch.hook_args module

      * syft.frameworks.torch.overload_torch module

      * syft.frameworks.torch.torch_attributes module

      * Module contents

  * Module contents

* syft.workers package

  * Submodules

  * syft.workers.abstract module

  * syft.workers.base module

  * syft.workers.plan module

  * syft.workers.virtual module

  * syft.workers.websocket_client module

  * syft.workers.websocket_server module

  * Module contents


## Submodules

## syft.codes module


#### class syft.codes.MSGTYPE()
Bases: `object`


#### CMD( = 1)

#### EXCEPTION( = 5)

#### GET_SHAPE( = 7)

#### IS_NONE( = 6)

#### OBJ( = 2)

#### OBJ_DEL( = 4)

#### OBJ_REQ( = 3)

#### SEARCH( = 8)
## syft.exceptions module

Specific Pysyft exceptions.


#### exception syft.exceptions.CannotRequestTensorAttribute()
Bases: `Exception`

Raised when .get() is called on a pointer which points to an attribute of
another tensor.


#### exception syft.exceptions.CompressionNotFoundException()
Bases: `Exception`

Raised when a non existent compression/decompression scheme is requested.


#### exception syft.exceptions.PureTorchTensorFoundError()
Bases: `BaseException`

Exception raised for errors in the input.
This error is used in a recursive analysis of the args provided as an
input of a function, to break the recursion if a TorchTensor is found
as it means that _probably_ all the tensors are pure torch tensor and
the function can be applied natively on this input.


#### expression -- input expression in which the error occurred()

#### message -- explanation of the error()

#### exception syft.exceptions.RemoteTensorFoundError(pointer)
Bases: `BaseException`

Exception raised for errors in the input.
This error is used in a context similar to PureTorchTensorFoundError but
to indicate that a Pointer to a remote tensor was found  in the input
and thus that the command should be send elsewhere. The pointer retrieved
by the error gives the location where the command should be sent.


#### expression -- input expression in which the error occurred()

#### message -- explanation of the error()

#### exception syft.exceptions.ResponseSignatureError(ids_generated=None)
Bases: `Exception`

Raised when the return of a hooked function is not correctly predicted
(when defining in advance ids for results)


#### exception syft.exceptions.TensorsNotCollocatedException(tensor_a, tensor_b, attr='a method')
Bases: `Exception`

Raised when a command is executed on two tensors which are not
on the same machine. The goal is to provide as useful input as possible
to help the user identify which tensors are where so that they can debug
which one needs to be moved.


#### exception syft.exceptions.WorkerNotFoundException()
Bases: `Exception`

Raised when a non-existent worker is requested.


#### syft.exceptions.route_method_exception(exception, self, args, kwargs)
## syft.grid module


#### class syft.grid.VirtualGrid(\*workers)
Bases: `object`


#### search(\*query, verbose=True, return_counter=True)
Searches over a collection of workers, returning pointers to the results
grouped by worker.

## syft.serde module

This file exists to provide one common place for all serialization to occur
regardless of framework. As msgpack only supports basic types and binary formats
every type must be first be converted to one of these types. Thus, we’ve split our
functionality into three steps. When converting from a PySyft object (or collection
of objects) to an object to be sent over the wire (a message), those three steps
are (in order):

1. Simplify - converts PyTorch objects to simple Python objects (using pickle)

1. Serialize - converts Python objects to binary

1. Compress - compresses the binary (Now we’re ready send!)

Inversely, when converting from a message sent over the wire back to a PySyft
object, the three steps are (in order):

1. Decompress - converts compressed binary back to decompressed binary

1. Deserialize - converts from binary to basic python objects

1. Detail - converts some basic python objects back to PyTorch objects (Tensors)

Furthermore, note that there is different simplification/serialization logic
for objects of different types. Thus, instead of using if/else logic, we have
global dictionaries which contain functions and Python types as keys. For
simplification logic, this dictionary is called “simplifiers”. The keys
are the types and values are the simplification logic. For example,
simplifiers[tuple] will return the function which knows how to simplify the
tuple type. The same is true for all other simplifier/detailer functions.

By default, the simplification/detail operations expect Torch tensors. If the setup requires other
serialization process, it can override the functions _serialize_tensor and _deserialize_tensor

By default, we serialize using msgpack and compress using lz4.
If different compressions are required, the worker can override the function _apply_compress_scheme


#### syft.serde.apply_lz4_compression(decompressed_input_bin)
Apply LZ4 compression to the input

:param : param decompressed_input_bin: the binary to be compressed
:param : return: a tuple (compressed_result, LZ4)


#### syft.serde.apply_no_compression(decompressed_input_bin)
No compression is applied to the input

:param : param decompressed_input_bin: the binary
:param : return: a tuple (the binary, LZ4)


#### syft.serde.apply_zstd_compression(decompressed_input_bin)
Apply ZSTD compression to the input

:param : param decompressed_input_bin: the binary to be compressed
:param : return: a tuple (compressed_result, ZSTD)


#### syft.serde.deserialize(binary: bin, worker: syft.workers.abstract.AbstractWorker = None, detail=True)
This method can deserialize any object PySyft needs to send or store.

This is the high level function for deserializing any object or collection
of objects which PySyft has sent over the wire or stored. It includes three
steps, Decompress, Deserialize, and Detail as described inline below.


* **Parameters**

    * **binary** (*bin*) – the serialized object to be deserialized.

    * **worker** (*AbstractWorker*) – the worker which is acquiring the message content,
      for example used to specify the owner of a tensor received(not obvious
      for virtual workers)

    * **detail** (*bool*) – there are some cases where we need to perform the decompression
      and deserialization part, but we don’t need to detail all the message.
      This is the case for Plan workers for instance



* **Returns**

    the deserialized form of the binary input.



* **Return type**

    object



#### syft.serde.numpy_tensor_deserializer(tensor_bin)
“Strategy to deserialize a binary input in npy format into a Torch tensor


#### syft.serde.numpy_tensor_serializer(tensor: torch.Tensor)
Strategy to serialize a tensor using numpy npy format.
If tensor requires to calculate gradients, it will detached.


#### syft.serde.serialize(obj: object, simplified=False)
This method can serialize any object PySyft needs to send or store.

This is the high level function for serializing any object or collection
of objects which PySyft needs to send over the wire. It includes three
steps, Simplify, Serialize, and Compress as described inline below.


* **Parameters**

    * **obj** (*object*) – the object to be serialized

    * **simplified** (*bool*) – in some cases we want to pass in data which has
      already been simplified - in which case we must skip double
      simplification - which would be bad…. so bad… so… so bad



* **Returns**

    the serialized form of the object.



* **Return type**

    binary



#### syft.serde.torch_tensor_deserializer(tensor_bin)
“Strategy to deserialize a binary input using Torch load


#### syft.serde.torch_tensor_serializer(tensor)
Strategy to serialize a tensor using Torch saver

## Module contents

Some syft imports…


#### class syft.TorchHook(torch, local_worker: syft.workers.base.BaseWorker = None, is_client: bool = True, verbose: bool = True)
Bases: `object`

A Hook which Overrides Methods on PyTorch Tensors.

The purpose of this class is to:

    * extend torch methods to allow for the moving of tensors from one

    worker to another.
    \* override torch methods to execute commands on one worker that are
    called on tensors controlled by the local worker.

This class is typically the first thing you will initialize when using
PySyft with PyTorch because it is responsible for augmenting PyTorch with
PySyft’s added functionality (such as remote execution).


* **Parameters**

    * **local_worker** – An optional BaseWorker instance that lets you provide a
      local worker as a parameter which TorchHook will assume to be the
      worker owned by the local machine. If you leave it empty,
      TorchClient will automatically initialize a
      `workers.VirtualWorker` under the assumption you’re looking
      to do local experimentation or development.

    * **is_client** – An optional boolean parameter (default True), indicating
      whether TorchHook is being initialized as an end-user client.This
      can impact whether or not variables are deleted when they fall out
      of scope. If you set this incorrectly on a end user client, Tensors
      and Variables will never be deleted. If you set this incorrectly on
      a remote machine (not a client), tensors will not get saved. It’s
      really only important if you’re not initializing the local worker
      yourself.

    * **verbose** – An optional boolean parameter (default True) to indicate
      whether or not to print the operations as they occur.

    * **queue_size** – An integer optional parameter (default 0) to specify the
      max length of the list that stores the messages to be sent.


### Example

```python
>>> import syft as sy
>>> hook = sy.TorchHook()
Hooking into Torch...
Overloading Complete.
>>> x = sy.Tensor([-2,-1,0,1,2,3])
>>> x
-2
-1
0
1
2
3
[syft.core.frameworks.torch.tensor.FloatTensor of size 6]
```


#### get_hooked_additive_shared_method(attr)
Hook a method to send it multiple recmote workers


* **Parameters**

    **attr** (*str*) – the method to hook



* **Returns**

    the hooked method



#### get_hooked_func(attr)
Hook a function in order to inspect its args and search for pointer
or other syft tensors.
- Calls to this function with normal tensors or numbers / string trigger

> usual behaviour

* Calls with pointers send the command to the location of the pointer(s)

* Calls with syft tensor will in the future trigger specific behaviour


* **Parameters**

    **attr** (*str*) – the method to hook



* **Returns**

    the hooked method



#### get_hooked_method(method_name)
Hook a method in order to replace all args/kwargs syft/torch tensors with
their child attribute if they exist
If so, forward this method with the new args and new self, get response
and “rebuild” the torch tensor wrapper upon all tensors found
If not, just execute the native torch methodn


* **Parameters**

    **attr** (*str*) – the method to hook



* **Returns**

    the hooked method



#### get_hooked_multi_pointer_method(attr)
Hook a method to send it multiple recmote workers


* **Parameters**

    **attr** (*str*) – the method to hook



* **Returns**

    the hooked method



#### get_hooked_pointer_method(attr)
Hook a method to send it to remote worker


* **Parameters**

    **attr** (*str*) – the method to hook



* **Returns**

    the hooked method



#### get_hooked_syft_method(attr)
Hook a method in order to replace all args/kwargs syft/torch tensors with
their child attribute, forward this method with the new args and new self,
get response and “rebuild” the syft tensor wrapper upon all tensors found


* **Parameters**

    **attr** (*str*) – the method to hook



* **Returns**

    the hooked method



#### class syft.VirtualWorker(hook: syft.frameworks.torch.hook.TorchHook, id: Union[int, str] = 0, data: Union[List, tuple] = None, is_client_worker: bool = False, log_msgs: bool = False, verbose: bool = False)
Bases: `syft.workers.base.BaseWorker`


#### class syft.Plan(hook, owner, name='', \*args, \*\*kwargs)
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

#### class syft.LoggingTensor(owner=None, id=None, tags=None, description=None)
Bases: `syft.frameworks.torch.tensors.interpreters.abstract.AbstractTensor`


#### add(\*args, \*\*kwargs)

#### manual_add(\*args, \*\*kwargs)
Here is the version of the add method without the decorator: as you can see
it is much more complicated. However you might need sometimes to specify
some particular behaviour: so here what to start from :)


#### classmethod on_function_call(command)
Override this to perform a specific action for each call of a torch
function with arguments containing syft tensors of the class doing
the overloading


#### torch( = <syft.frameworks.torch.overload_torch.Module object>)

#### class syft.PointerTensor(location: BaseWorker = None, id_at_location: Union[str, int] = None, owner: BaseWorker = None, id: Union[str, int] = None, garbage_collect_data: bool = True, shape: torch.Size = None, point_to_attr: str = None, tags: List[str] = None, description: str = None)
Bases: `syft.frameworks.torch.tensors.interpreters.abstract.AbstractTensor`

A pointer to another tensor.

A PointerTensor forwards all API calls to the remote.PointerTensor objects
point to tensors (as their name implies). They exist to mimic the entire
API of a normal tensor, but instead of computing a tensor function locally
(such as addition, subtraction, etc.) they forward the computation to a
remote machine as specified by self.location. Specifically, every
PointerTensor has a tensor located somewhere that it points to (they should
never exist by themselves). Note that PointerTensor objects can point to
both torch.Tensor objects AND to other PointerTensor objects. Furthermore,
the objects being pointed to can be on the same machine or (more commonly)
on a different one. Note further that a PointerTensor does not know the
nature how it sends messages to the tensor it points to (whether over
socket, http, or some other protocol) as that functionality is abstracted
in the BaseWorker object in self.location.

### Example

```python
>>> import syft as sy
>>> hook = sy.TorchHook()
>>> bob = sy.VirtualWorker(id="bob")
>>> x = sy.Tensor([1,2,3,4,5])
>>> y = sy.Tensor([1,1,1,1,1])
>>> x_ptr = x.send(bob) # returns a PointerTensor, sends tensor to Bob
>>> y_ptr = y.send(bob) # returns a PointerTensor, sends tensor to Bob
>>> # executes command on Bob's machine
>>> z_ptr = x_ptr + y_ptr
```


#### attr(attr_name)

#### data()

#### classmethod find_a_pointer(command)
Find and return the first pointer in the args object, using a trick
with the raising error RemoteTensorFoundError


#### get(deregister_ptr: bool = True)
Requests the tensor/chain being pointed to, be serialized and return

Since PointerTensor objects always point to a remote tensor (or chain
of tensors, where a chain is simply a linked-list of tensors linked via
their .child attributes), this method will request that the tensor/chain
being pointed to be serialized and returned from this function.

**NOTE**: This will typically mean that the remote object will be
removed/destroyed. To just bring a copy back to the local worker,
call .copy() before calling .get().


* **Parameters**

    **deregister_ptr** (*bool**, **optional*) – this determines whether to
    deregister this pointer from the pointer’s owner during this
    method. This defaults to True because the main reason people use
    this method is to move the tensor from the remote machine to the
    local one, at which time the pointer has no use.



* **Returns**

    An AbstractTensor object which is the tensor (or chain) that this
    object used to point to on a remote machine.


TODO: add param get_copy which doesn’t destroy remote if true.


#### get_shape()
Request information about the shape to the remote worker


#### grad()

#### classmethod handle_func_command(command)
Receive an instruction for a function to be applied on a Pointer,
Get the remote location to send the command, send it and get a
pointer to the response, return.
:param command: instruction of a function command: (command name,
None, arguments[, kwargs])
:return: the response of the function command


#### is_none()

#### shape()
This method returns the shape of the data being pointed to.
This shape information SHOULD be cached on self._shape, but
occasionally this information may not be present. If this is the
case, then it requests the shape information from the remote object
directly (which is inefficient and should be avoided).


#### share(\*args, \*\*kwargs)
Send a command to remote worker to additively share a tensor


* **Returns**

    A pointer to an AdditiveSharingTensor



#### class syft.VirtualGrid(\*workers)
Bases: `object`


#### search(\*query, verbose=True, return_counter=True)
Searches over a collection of workers, returning pointers to the results
grouped by worker.
