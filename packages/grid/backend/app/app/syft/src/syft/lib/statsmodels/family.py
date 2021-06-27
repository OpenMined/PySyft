# stdlib
from typing import Dict
from typing import Type

# third party
import statsmodels

# syft relative
from ...generate_wrapper import GenerateWrapper
from ...lib.python.primitive_factory import PrimitiveFactory
from ...lib.python.string import String
from ...proto.lib.statsmodels.family_pb2 import FamilyProto

FAMILY_2_STR: Dict[Type[statsmodels.genmod.families.family.Family], str] = {
    statsmodels.genmod.families.family.Binomial: "Binomial",
    statsmodels.genmod.families.family.Gamma: "Gamma",
    statsmodels.genmod.families.family.Gaussian: "Gaussian",
    statsmodels.genmod.families.family.InverseGaussian: "InverseGaussian",
    statsmodels.genmod.families.family.NegativeBinomial: "NegativeBinomial",
    statsmodels.genmod.families.family.Poisson: "Poisson",
    statsmodels.genmod.families.family.Tweedie: "Tweedie",
}
LINK_2_STR: Dict[Type[statsmodels.genmod.families.family.Family], str] = {
    statsmodels.genmod.families.links.log: "log",
    statsmodels.genmod.families.links.logit: "logit",
    statsmodels.genmod.families.links.cauchy: "cauchy",
    statsmodels.genmod.families.links.cloglog: "cloglog",
    statsmodels.genmod.families.links.identity: "identity",
    statsmodels.genmod.families.links.inverse_power: "inverse_power",
    statsmodels.genmod.families.links.inverse_squared: "inverse_squared",
    statsmodels.genmod.families.links.nbinom: "nbinom",
    statsmodels.genmod.families.links.probit: "probit",
    statsmodels.genmod.families.links.Power: "Power",
    statsmodels.genmod.families.links.NegativeBinomial: "NegativeBinomial",
    statsmodels.genmod.families.links.CDFLink: "CDFLink",
    statsmodels.genmod.families.links.sqrt: "sqrt",
}

STR_2_FAMILY: Dict[str, Type[statsmodels.genmod.families.family.Family]] = {
    v: k for k, v in FAMILY_2_STR.items()
}
STR_2_LINK: Dict[str, Type[statsmodels.genmod.families.family.Family]] = {
    v: k for k, v in LINK_2_STR.items()
}


def object2proto(obj: Type[statsmodels.genmod.families.family.Family]) -> FamilyProto:
    family_name = FAMILY_2_STR[type(obj)]
    link_name = LINK_2_STR[type(obj.link)]
    family_name_prim = PrimitiveFactory.generate_primitive(value=family_name)
    link_name_prim = PrimitiveFactory.generate_primitive(value=link_name)
    family_name_proto = family_name_prim._object2proto()
    link_name_proto = link_name_prim._object2proto()

    return FamilyProto(family=family_name_proto, link=link_name_proto)


def proto2object(proto: FamilyProto) -> Type[statsmodels.genmod.families.family.Family]:
    family_name = str(String._proto2object(proto=proto.family))
    link_name = str(String._proto2object(proto=proto.link))
    obj = STR_2_FAMILY[family_name](link=STR_2_LINK[link_name])
    return obj


for fam in FAMILY_2_STR.keys():
    GenerateWrapper(
        wrapped_type=fam,
        import_path="statsmodels.genmod.families.family" + fam.__class__.__name__,
        protobuf_scheme=FamilyProto,
        type_object2proto=object2proto,
        type_proto2object=proto2object,
    )
