# third party
from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper

# syft relative
from ...generate_wrapper import GenerateWrapper
from ...lib.python.primitive_factory import PrimitiveFactory
from ...lib.python.string import String
from ...proto.lib.statsmodels.results_pb2 import ResultsProto


def object2proto(obj: GLMResultsWrapper) -> ResultsProto:
    summary = obj.summary().as_csv()
    summary_prim = PrimitiveFactory.generate_primitive(value=summary)
    summary_proto = summary_prim._object2proto()
    return ResultsProto(summary=summary_proto)


def proto2object(proto: ResultsProto) -> str:
    return str(String._proto2object(proto.summary))


GenerateWrapper(
    wrapped_type=GLMResultsWrapper,
    import_path="statsmodels.genmod.generalized_linear_model.GLMResultsWrapper",
    protobuf_scheme=ResultsProto,
    type_object2proto=object2proto,
    type_proto2object=proto2object,
)
