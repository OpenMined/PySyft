# syft.frameworks.torch.tensors.decorators package

## Submodules

## syft.frameworks.torch.tensors.decorators.logging module


#### class syft.frameworks.torch.tensors.decorators.logging.LoggingTensor(owner=None, id=None, tags=None, description=None)
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
## Module contents
