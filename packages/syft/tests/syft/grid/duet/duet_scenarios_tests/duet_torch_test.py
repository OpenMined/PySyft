# stdlib
import sys
import time


def do_test(port: int) -> None:
    # third party
    import torch

    # syft absolute
    import syft as sy
    if sy.experimental_flags.flags.TEST_FLIGHT:
        flight_test_params = [True, False]
    else:
        flight_test_params = [False]
    
    sy.logger.add(sys.stderr, "ERROR")

    duet = sy.launch_duet(loopback=True, network_url=f"http://127.0.0.1:{port}/")
    duet.requests.add_handler(action="accept")

    for flight_status in flight_test_params:
        sy.experimental_flags.flags.APACHE_ARROW_FLIGHT_CHANNEL = flight_status
        t = torch.randn(20).reshape(4, 5)
        _ = t.send(duet, pointable=True)

    sy.core.common.event_loop.loop.run_forever()


def ds_test(port: int) -> None:
    # third party
    import torch

    # syft absolute
    import syft as sy
    if sy.experimental_flags.flags.TEST_FLIGHT:
        count = 2
    else:
        count = 1
    
    sy.logger.add(sys.stderr, "ERROR")

    duet = sy.join_duet(loopback=True, network_url=f"http://127.0.0.1:{port}/")

    time.sleep(3)

    for item in range(count):
        data = duet.store[0].get(request_block=True, delete_obj=False)

        assert data.shape == torch.Size([4, 5]), data.shape


test_scenario_torch_tensor_sanity = (
    "test_scenario_torch_tensor_sanity",
    do_test,
    ds_test,
)
