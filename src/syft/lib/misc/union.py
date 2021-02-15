# stdlib
from typing import Any
from typing import Dict
from typing import List
from typing import Set
from typing import Tuple


class IndexableTrait(type):
    """
    IndexableTrait is used as a metaclass for our UnionGenerator to be able to
    set __getitem__ on the class to enable the UnionGenerator[...] syntax.
    """

    def __getitem__(self, union_types: Tuple[str, ...]) -> str:
        return UnionGenerator.from_qualnames(list(union_types))


class UnionGenerator(type, metaclass=IndexableTrait):
    """
    UnionGenerator is a metaclass used to generate on the spot union types
    from their qualnames.

    Eg. UnionGenerator["syft.lib.python.Int", "syft.lib.python.Float"] will generate
    in the current module the FloatIntUnion type that will have no functions added
    to the type itself until the AST is finished to be generated. If new types of
    unions have to be generated afterwards:
    1. you might be using the ast in a wrong way.
    2. if you know what you are doing, you have to regenerate the ast.
    """

    SUFFIX_UNION = "Union"

    def __new__(cls, name: str, bases: tuple, dct: dict) -> Any:
        dct["__qualname__"] = "syft.lib.misc.union." + name
        dct["__name__"] = name
        new_type = super().__new__(cls, name, bases, dct)
        # insert our new union type in the global scope of the module
        globals()[new_type.__name__] = new_type
        return new_type

    @staticmethod
    def from_qualnames(union_types: List[str]) -> str:
        # sorting to not generate the same permutation of the types
        union_types = sorted(union_types)

        name = "".join([union_type.split(".")[-1] for union_type in union_types])
        name += UnionGenerator.SUFFIX_UNION

        # searching in the cache to not solve the union type multiple times
        if name in union_cache:
            return name

        target_type = UnionGenerator(name, tuple(), {})

        # adding the type and it's union types to a lazy dict that will be solved
        # when the ast is created.
        lazy_pairing[target_type] = union_types
        qualname = target_type.__qualname__
        union_cache.add(qualname)
        return qualname


# store here already solved union types
union_cache: Set[str] = set()

# store here union types to be solved after the AST has been created and we
# can generate our misc types based on them.
lazy_pairing: Dict[UnionGenerator, List[str]] = {}
