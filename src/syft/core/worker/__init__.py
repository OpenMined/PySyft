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

* :py:mod:`syft.core.worker.domain.Domain` - this API is the interface to a collection of datasets owned by a single \
entity.
* :py:mod:`syft.core.worker.worker.Worker` - this API is the interface to a remote machine within a data owner's \
domain.
* :py:mod:`syft.core.worker.client.Client` - This API is the interface a data scientist uses to interact with a worker \
within a domain

So, a domain would be something like "Big Fancy Hospital" and a worker would be a single machine within that hospital
which a Data Scientist can use to process some data. A domain will have many workers, each of which could be serving
a different Data Scientist. Alternatively, one Data Scientist could be using multiple workers (such as if they have
very computationally expensive programs to run and they want to run them in parallel).

However, a Data Scientist wouldn't interact with these remote Domain and Worker APIs directly. Instead, they use their
:py:mod:`syft.core.worker.client.Client` api which has lots of convenience functions for interacting with remote
machines held within remote domains.

The :py:mod:`syft.core.worker.client.Client` API will send :py:mod:`syft.core.message.message.Message` objects to the
:py:mod:`syft.core.worker.domain.Domain` API which will handle some messages itself and, if appropriate, forward other
messages to the appropriate :py:mod:`syft.core.worker.worker.Worker` for execution on real objects.

However, you will find that Domain, Worker, and Client all have missing functionality, namely how to send a message!
This might seem like a critical oversight, but it's actually an important abstraction. These classes are merely
interfaces for the API language that all Domain, Worker, and Client objects should use. This language is universal
regardless of whether two phones, two hospitals, or two satellites are talking to each other! However, the exact
transport protocol for _how_ messages are sent we leave up to the _specific instance of a worker.

The primary instance of Domain/Worker/Client that we use for development, testing, and learning is the
VirtualDomain/VirtualWorker/VirtualClient instance. More on that in a moment...

If you're learning Syft, the next step you should take is to jump into the :py:mod:`syft.core.worker.domain.Domain`
class...


"""

from .service.worker_service import WorkerService
from typing import Dict

message_service_mapping: Dict[str, WorkerService] = {}

from . import worker
from . import service