def do_test(port: int) -> None:
    # syft absolute
    import syft as sy

    duet = sy.launch_duet(loopback=True, network_url=f"http://127.0.0.1:{port}/")
    _ = sy.lib.python.List([1, 2, 3]).send(duet)

    sy.core.common.event_loop.loop.run_forever()


def ds_test(port: int) -> None:
    # syft absolute
    import syft as sy

    _ = sy.join_duet(loopback=True, network_url=f"http://127.0.0.1:{port}/")


test_scenario_init = ("test_scenario_init", do_test, ds_test)
