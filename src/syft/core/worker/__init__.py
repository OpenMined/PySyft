# -*- coding: utf-8 -*-
"""
Welcome to the :py:mod:`syft.core.worker` module! This is a good place to begin your education of
what Syft is and how it works.

At it's core, Syft is a set of libraries which allows you to perform data processing on data you
cannot see. We have two core "personas" within the Syft ecosystem, the "data owner" and the "data scientist"
(who is sometimes referred to as the "model owner"). The data owner has data to protect, and the data scientist
wants to answer a question using data owned by one or more data owners.

Note that a data owner could be a consumer who has data on their phone, a hospital with medical records, or
even a Raspberry PI floating in the middle of the Pacific Ocean! It just represents a collection of data
within a Data Owner's domain of ownership. As such, there are three core abstractions you should know about:

* :py:mod:`syft.core.worker.domain` - this API is the interface to a collection of datasets owned by a single entity.
* :py:mod:`syft.core.worker.worker` - this API is the interface to a remote machine within a data owner's domain.
* :py:mod:`syft.core.worker.client` - This API is the interface a data scientist uses to interact with a worker within\
 a domain

So, a domain would be something like "Big Fancy Hospital" and a worker would be a single machine within that hospital
which a Data Scientist can use to process some data. A domain will have many workers, each of which could be serving
a different Data Scientist. Alternatively, one Data Scientist could be using multiple workers (such as if they have
very computationally expensive programs to run and they want to run them in parallel).

"""


from .virtual.virtual_worker import VirtualWorker


def create_virtual_workers(*args):
    clients = list()
    for worker_name in args:
        clients.append(VirtualWorker(worker_name).get_client())
    return clients
