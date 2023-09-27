# stdlib
from time import sleep

# syft absolute
import syft as sy
from syft import ActionObject
from syft import syft_function
from syft import syft_function_single_use

# node = sy.orchestra.launch(name="test-domain-helm", port=8800, dev_mode=True, reset=True, tail=True)


# client = sy.login(port=8800, email="info@openmined.org", password="changethis")


node = sy.orchestra.launch(
    name="test-domain-helm2", dev_mode=True, reset=True, n_consumers=0
)
client = node.login(email="info@openmined.org", password="changethis")

node.python_node.queue_config

# syft absolute
from syft.service.queue.zmq_queue import ZMQClientConfig
from syft.service.queue.zmq_queue import ZMQQueueConfig

worker_node = sy.orchestra.launch(
    name="worker-node-helm1",
    dev_mode=True,
    reset=True,
    n_consumers=3,
    queue_config=ZMQQueueConfig(client_config=ZMQClientConfig(create_producer=False)),
)


# from gevent import monkey

# monkey.patch_all(thread=True)

# type(node.python_node.queue_manager.consumers["api_call"])

# node.python_node.queue_manager.consumers["api_call"][0].address


# Setup syft functions

## Dataset

x = ActionObject.from_obj([1, 2])
x_ptr = x.send(client)

## Batch function


@syft_function()
def process_batch(batch):
    # takes 30 hours normally
    print(f"starting batch {batch}")
    # stdlib
    from time import sleep

    sleep(1)
    print("done")
    return batch + 1


client.code.submit(process_batch)

## Main function


@syft_function_single_use(x=x_ptr)
def process_all(domain, x):
    jobs = []
    print("Launching jobs")
    for elem in x:
        # We inject a domain object in the scope
        batch_job = domain.launch_job(process_batch, batch=elem)
        jobs += [batch_job]
    print("starting aggregation")
    print("Done")
    results = [x.wait().get() for x in jobs]
    #     return 1
    return sum(results)


# Approve & run

client.code.request_code_execution(process_all)
client.requests[-1].approve()

job = client.code.process_all(x=x_ptr, blocking=False)
sleep(1)

job.wait().get()
