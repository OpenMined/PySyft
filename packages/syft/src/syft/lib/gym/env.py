# third party
import gym

# relative
from ...generate_wrapper import GenerateWrapper
from ...proto.lib.gym.env_pb2 import Env as Env_PB

gym_env_type = type(gym.Env())


def object2proto(obj: gym.Env) -> Env_PB:
    return Env_PB(id=obj.unwrapped.spec.id)


def proto2object(proto: Env_PB) -> gym.Env:
    return gym.make(proto.id)


GenerateWrapper(
    wrapped_type=gym_env_type,
    import_path="gym.Env",
    protobuf_scheme=Env_PB,
    type_object2proto=object2proto,
    type_proto2object=proto2object,
)
