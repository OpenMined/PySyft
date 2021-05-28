# third party
import gym
import numpy as np
import pytest

# syft absolute
import syft as sy
from syft.experimental_flags import flags
from syft.lib.gym.env import object2proto
from syft.lib.gym.env import proto2object

sy.load("gym")
sy.load("numpy")


@pytest.mark.vendor(lib="gym")
@pytest.mark.parametrize("arrow_backend", [False, True])
def test_remote_gym(arrow_backend: bool, root_client: sy.VirtualMachineClient) -> None:
    flags.APACHE_ARROW_TENSOR_SERDE = arrow_backend

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


@pytest.mark.vendor(lib="gym")
def test_protobuf(root_client: sy.VirtualMachineClient) -> None:
    env = gym.make("CartPole-v0")
    env.seed(42)
    pb = object2proto(env)
    deserialized_env = proto2object(pb)
    assert deserialized_env.unwrapped.spec.id == "CartPole-v0"

    env.seed(42)
    deserialized_env.seed(42)

    initial_state = env.reset()
    deserialized_initial_state = deserialized_env.reset()
    assert np.array_equal(initial_state, deserialized_initial_state)

    state, reward, done, info = env.step(0)
    (
        deserialized_state,
        deserialized_reward,
        deserialized_done,
        deserialized_info,
    ) = deserialized_env.step(0)
    assert np.array_equal(state, deserialized_state)
    assert reward == deserialized_reward
    assert done == deserialized_done
    assert info == deserialized_info
