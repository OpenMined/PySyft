# PySyft

A library for computing on data you do not own and cannot see.


## Description

Most software libraries let you compute over information you own and see inside of machines you control. However, this means that you cannot compute on information without first obtaining (at least partial) ownership of that information. It also means that you cannot compute using machines without first obtaining control over those machines. This is very limiting to human collaboration in all areas of life and systematically drives the centralization of data, because you cannot work with a bunch of data without first putting it all in one (central) place.

The Syft ecosystem seeks to change this system, allowing you to write software which can compute over information you do not own on machines you do not have (general) control over.

This library is the centerpiece of the Syft ecosystem. It has two primary purposes. You can either use PySyf to:

1) *Dynamic:* Directly compute over data you cannot see.
2) *Static:* Create static graphs of computation which can be deployed/scaled at a later date on different compute.

The Syft ecosystem includes libraries which allow for communication with and computation over a variety of runtimes:

- KotlinSyft (Android)
- SwiftSyft (iOS)
- syft.js (Web & Mobile)
- PySyft (Python)

However, the Syft ecosystem only focuses on consistent object serialization/deserialization, core abstractions, and algorithm design/execution across these languages. These libraries alone will not connect you with data in the real world. The Syft ecosystem is supported by the Grid ecosystem, which focuses on deployment, scalability, and other additional concerns around running real-world systems to compute over and process data (such as data compliance web applications). Syft is the library that defines objects, abstractions, and algorithms. Grid is the platform which lets you deploy them within a real institution (or on the open internet, but we don't yet recommend this). The Grid ecosystem includes:

- GridNetwork - think of this like DNS for private data. It helps you find remote data assets so that you can compute with them.
- PyGrid - This is the gateway to an organization's data, responsible for permissions, load balancing, and governance.
- GridNode - This is an individual worker within an organization's datacenter, running executions requested by external parties.
- GridMonitor - This is a UI which allows an institution to introspect and control their PyGrid worker and the GridNodes it manages.

## Note

This project has been set up using PyScaffold 3.2.3. For details and usage
information on PyScaffold see https://pyscaffold.org/.
