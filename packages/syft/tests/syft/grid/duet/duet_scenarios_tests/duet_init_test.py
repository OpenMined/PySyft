def do_test(port: int) -> None:
    # syft absolute
    import syft as sy

    duet = sy.launch_duet(loopback=True, network_url=f"http://127.0.0.1:{port}/")
    _ = sy.lib.python.List([1, 2, 3]).send(duet)

    if sy.experimental_flags.flags.TEST_FLIGHT:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            assert s.connect_ex(('localhost', sy.experimental_flags.flags.FLIGHT_CHANNEL_PORT)) != 0

    sy.core.common.event_loop.loop.run_forever()


def ds_test(port: int) -> None:
    # syft absolute
    import syft as sy

    _ = sy.join_duet(loopback=True, network_url=f"http://127.0.0.1:{port}/")


test_scenario_init = ("test_scenario_init", do_test, ds_test)
