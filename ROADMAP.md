This file contains a high level roadmap plan for upcoming releases of PySyft.

# Version 0.3

The main priority for this version is to create a single, standardized API and serialization format for PySyft which can execute Plans (lists of messages) generated from PySyft across multiple platforms: Python, Android, iOS, and Javascript.

The second big priority is the refactoring and standardization of tensor abstractions. Previously, we had started to develop bad tensor habits such as folding too much functionality into single tensor type classes (monolith tensors) as well as implementing with NumPy backends instead of PyTorch backends.- 

- Msgpack -> Protobuf for all object serialization

- Upgrade Plan object
  - support two kinds of plans: "list of operations" style and JIT style.
  - ensure that "list of operations" style supports special functionality such as plans within plans and .send() functionality.
  
  - extend Plans to facilitate hooking at a specific "level" in the class abstraction hierarchy. Or it could simply be the "lowest" level by default.

- Finish PromiseTensor object

- Automatic conversion from PromiseTensor -> Protocol object

# Version 0.4

