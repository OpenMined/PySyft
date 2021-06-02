# third party
import pytest

# syft absolute
import syft as sy


@pytest.mark.vendor(lib="gym")
def test_remote_gym(root_client: sy.VirtualMachineClient) -> None:
    sy.load("gym")
    sy.load("numpy")

    # third party
    import gym
    import numpy as np

    remote_gym = root_client.gym

    env = gym.make("CartPole-v0")
    remote_env = remote_gym.make("CartPole-v0")

    env.seed(42)
    remote_env.seed(42)
    assert remote_env.__name__ == "TimeLimitPointer"

    initial_state = env.reset()
    remote_initial_state = remote_env.reset().get()
    assert np.array_equal(initial_state, remote_initial_state)

    state, reward, done, info = env.step(0)
    remote_state, remote_reward, remote_done, remote_info = remote_env.step(0).get()

    assert np.array_equal(state, remote_state)
    assert reward == remote_reward
    assert done == remote_done
    assert info == remote_info
