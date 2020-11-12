# stdlib
from random import randint
from typing import List
from typing import Union

# third party
from forbiddenfruit import curse
from loguru import logger
from nacl.signing import SigningKey
from nacl.signing import VerifyKey

# syft absolute
# breaking convention here because index_globals needs
# the full syft name to be present.
import syft

# syft relative
from .core.common.group import VerifyAll
from .decorators.syft_decorator_impl import syft_decorator


@syft_decorator(typechecking=True)
def get_subclasses(obj_type: type) -> List[type]:
    """Recusively generate the list of all classes within the sub-tree of an object

    As a paradigm in Syft, we often allow for something to be known about by another
    part of the codebase merely because it has subclassed a particular object. While
    this can be a big "magicish" it also can simplify future extensions and reduce
    the likelihood of small mistakes (if done right).

    This is a utility function which allows us to look for sub-classes and the sub-classes
    of those sub-classes etc. returning a full list of descendants of a class in the inheritance
    hierarchy.

    :param obj_type: the type we want to look for sub-classes of
    :type type:
    :returns: the list of subclasses of obj_type:
    :rtype: List[type]

    """

    classes = list()
    for sc in obj_type.__subclasses__():
        classes.append(sc)
        classes += get_subclasses(obj_type=sc)
    return classes


@syft_decorator(typechecking=True)
def index_modules(a_dict: object, keys: List[str]) -> object:
    """Recursively find a syft module from its path

    This is the recursive inner function of index_syft_by_module_name.
    See that method for a full description.

    :param a_dict: a module we're traversing
    :type a_dict: object
    :param keys: the list of string attributes we're using to traverse the module
    :type keys: List[str]
    :returns: a reference to the final object
    :rtype: object

    """

    if len(keys) == 0:
        return a_dict
    return index_modules(a_dict=a_dict.__dict__[keys[0]], keys=keys[1:])


@syft_decorator(typechecking=True)
def index_syft_by_module_name(fully_qualified_name: str) -> object:
    """Look up a Syft class/module/function from full path and name

    Sometimes we want to use the fully qualified name (such as one
    generated from the 'get_fully_qualified_name' method below) to
    fetch an actual reference. This is most commonly used in deserialization
    so that we can have generic protobuf objects which just have a string
    representation of the specific object it is meant to deserialize to.

    :param fully_qualified_name: the name of a module, class, or function
    :type fully_qualified_name: str
    :returns: a reference to the actual object at that string path
    :rtype: object

    """

    attr_list = fully_qualified_name.split(".")
    assert attr_list[0] == "syft"
    assert attr_list[1] == "core" or attr_list[1] == "lib" or attr_list[1] == "grid"
    return index_modules(a_dict=globals()["syft"], keys=attr_list[1:])


@syft_decorator(typechecking=True)
def get_fully_qualified_name(obj: object) -> str:
    """Return the full path and name of a class

    Sometimes we want to return the entire path and name encoded
    using periods. For example syft.core.common.message.SyftMessage
    is the current fully qualified path and name for the SyftMessage
    object.

    :param obj: the object we we want to get the name of
    :type obj: object
    :returns: the full path and name of the object
    :rtype: str

    """

    fqn = obj.__module__
    try:
        fqn += "." + obj.__class__.__name__
    except Exception as e:
        logger.error(f"Failed to get FQN: {e}")
    return fqn


@syft_decorator(typechecking=True)
def aggressive_set_attr(obj: object, name: str, attr: object) -> None:
    """Different objects prefer different types of monkeypatching - try them all"""

    try:
        setattr(obj, name, attr)
    except Exception:
        curse(obj, name, attr)


def obj2pointer_type(obj: object) -> type:
    try:
        fqn = get_fully_qualified_name(obj=obj)
    except Exception as e:
        # sometimes the object doesn't have a __module__ so you need to use the type
        # like: collections.OrderedDict
        logger.debug(
            f"Unable to get get_fully_qualified_name of {type(obj)} trying type. {e}"
        )
        if obj is None:
            fqn = "syft.lib.python._SyNone"
        else:
            fqn = get_fully_qualified_name(obj=type(obj))
    try:
        ref = syft.lib_ast(fqn, return_callable=True)
    except Exception:
        # try one more time by removing the class parent module name
        try:
            ref = syft.lib_ast(fqn, return_callable=True, obj_type=type(obj))
        except Exception as e:
            logger.critical(f"Cannot find {type(obj)} {fqn} in lib_ast. {e}")
        # TODO maybe return AnyPointer?

    return ref.pointer_type


@syft_decorator(typechecking=True)
def key_emoji(key: Union[bytes, SigningKey, VerifyKey, VerifyAll]) -> str:
    try:
        if isinstance(key, (bytes, SigningKey, VerifyKey)):
            hex_chars = bytes(key).hex()[-8:]
            return char_emoji(hex_chars=hex_chars)
    except Exception:
        pass
    return "ALL"


@syft_decorator(typechecking=True)
def char_emoji(hex_chars: str) -> str:
    base = ord("\U0001F642")
    hex_base = ord("0")
    code = 0
    for char in hex_chars:
        offset = ord(char)
        code += offset - hex_base
    return chr(base + code)


left_name = [
    "admiring",
    "adoring",
    "affectionate",
    "agitated",
    "amazing",
    "angry",
    "awesome",
    "beautiful",
    "blissful",
    "bold",
    "boring",
    "brave",
    "busy",
    "charming",
    "clever",
    "cool",
    "compassionate",
    "competent",
    "condescending",
    "confident",
    "cranky",
    "crazy",
    "dazzling",
    "determined",
    "distracted",
    "dreamy",
    "eager",
    "ecstatic",
    "elastic",
    "elated",
    "elegant",
    "eloquent",
    "epic",
    "exciting",
    "fervent",
    "festive",
    "flamboyant",
    "focused",
    "friendly",
    "frosty",
    "funny",
    "gallant",
    "gifted",
    "goofy",
    "gracious",
    "great",
    "happy",
    "hardcore",
    "heuristic",
    "hopeful",
    "hungry",
    "infallible",
    "inspiring",
    "interesting",
    "intelligent",
    "jolly",
    "jovial",
    "keen",
    "kind",
    "laughing",
    "loving",
    "lucid",
    "magical",
    "mystifying",
    "modest",
    "musing",
    "naughty",
    "nervous",
    "nice",
    "nifty",
    "nostalgic",
    "objective",
    "optimistic",
    "peaceful",
    "pedantic",
    "pensive",
    "practical",
    "priceless",
    "quirky",
    "quizzical",
    "recursing",
    "relaxed",
    "reverent",
    "romantic",
    "sad",
    "serene",
    "sharp",
    "silly",
    "sleepy",
    "stoic",
    "strange",
    "stupefied",
    "suspicious",
    "sweet",
    "tender",
    "thirsty",
    "trusting",
    "unruffled",
    "upbeat",
    "vibrant",
    "vigilant",
    "vigorous",
    "wizardly",
    "wonderful",
    "xenodochial",
    "youthful",
    "zealous",
    "zen",
]

right_name = [
    "albattani",
    "allen",
    "almeida",
    "antonelli",
    "agnesi",
    "archimedes",
    "ardinghelli",
    "aryabhata",
    "austin",
    "babbage",
    "banach",
    "banzai",
    "bardeen",
    "bartik",
    "bassi",
    "beaver",
    "bell",
    "benz",
    "bhabha",
    "bhaskara",
    "black",
    "blackburn",
    "blackwell",
    "bohr",
    "booth",
    "borg",
    "bose",
    "bouman",
    "boyd",
    "brahmagupta",
    "brattain",
    "brown",
    "buck",
    "burnell",
    "cannon",
    "carson",
    "cartwright",
    "carver",
    "cerf",
    "chandrasekhar",
    "chaplygin",
    "chatelet",
    "chatterjee",
    "chebyshev",
    "cohen",
    "chaum",
    "clarke",
    "colden",
    "cori",
    "cray",
    "curran",
    "curie",
    "darwin",
    "davinci",
    "dewdney",
    "dhawan",
    "diffie",
    "dijkstra",
    "dirac",
    "driscoll",
    "dubinsky",
    "easley",
    "edison",
    "einstein",
    "elbakyan",
    "elgamal",
    "elion",
    "ellis",
    "engelbart",
    "euclid",
    "euler",
    "faraday",
    "feistel",
    "fermat",
    "fermi",
    "feynman",
    "franklin",
    "gagarin",
    "galileo",
    "galois",
    "ganguly",
    "gates",
    "gauss",
    "germain",
    "goldberg",
    "goldstine",
    "goldwasser",
    "golick",
    "goodall",
    "gould",
    "greider",
    "grothendieck",
    "haibt",
    "hamilton",
    "haslett",
    "hawking",
    "hellman",
    "heisenberg",
    "hermann",
    "herschel",
    "hertz",
    "heyrovsky",
    "hodgkin",
    "hofstadter",
    "hoover",
    "hopper",
    "hugle",
    "hypatia",
    "ishizaka",
    "jackson",
    "jang",
    "jemison",
    "jennings",
    "jepsen",
    "johnson",
    "joliot",
    "jones",
    "kalam",
    "kapitsa",
    "kare",
    "keldysh",
    "keller",
    "kepler",
    "khayyam",
    "khorana",
    "kilby",
    "kirch",
    "knuth",
    "kowalevski",
    "lalande",
    "lamarr",
    "lamport",
    "leakey",
    "leavitt",
    "lederberg",
    "lehmann",
    "lewin",
    "lichterman",
    "liskov",
    "lovelace",
    "lumiere",
    "mahavira",
    "margulis",
    "matsumoto",
    "maxwell",
    "mayer",
    "mccarthy",
    "mcclintock",
    "mclaren",
    "mclean",
    "mcnulty",
    "mendel",
    "mendeleev",
    "meitner",
    "meninsky",
    "merkle",
    "mestorf",
    "mirzakhani",
    "moore",
    "morse",
    "murdock",
    "moser",
    "napier",
    "nash",
    "neumann",
    "newton",
    "nightingale",
    "nobel",
    "noether",
    "northcutt",
    "noyce",
    "panini",
    "pare",
    "pascal",
    "pasteur",
    "payne",
    "perlman",
    "pike",
    "poincare",
    "poitras",
    "proskuriakova",
    "ptolemy",
    "raman",
    "ramanujan",
    "ride",
    "montalcini",
    "ritchie",
    "rhodes",
    "robinson",
    "roentgen",
    "rosalind",
    "rubin",
    "saha",
    "sammet",
    "sanderson",
    "satoshi",
    "shamir",
    "shannon",
    "shaw",
    "shirley",
    "shockley",
    "shtern",
    "sinoussi",
    "snyder",
    "solomon",
    "spence",
    "stonebraker",
    "sutherland",
    "swanson",
    "swartz",
    "swirles",
    "taussig",
    "tereshkova",
    "tesla",
    "tharp",
    "thompson",
    "torvalds",
    "tu",
    "turing",
    "varahamihira",
    "vaughan",
    "visvesvaraya",
    "volhard",
    "villani",
    "wescoff",
    "wilbur",
    "wiles",
    "williams",
    "williamson",
    "wilson",
    "wing",
    "wozniak",
    "wright",
    "wu",
    "yalow",
    "yonath",
    "zhukovsky",
]


# we could replace these with some favorite AI / Privacy researches and engineers?
@syft_decorator(typechecking=True)
def random_name() -> str:
    left_i = randint(0, len(left_name) - 1)
    right_i = randint(0, len(right_name) - 1)
    return f"{left_name[left_i].capitalize()} {right_name[right_i].capitalize()}"
