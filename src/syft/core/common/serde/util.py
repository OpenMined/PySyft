from .serializable import Serializable
from ..lazy_structures import LazyDict
from ....decorators.syft_decorator_impl import syft_decorator
from ....util import index_syft_by_module_name
from ....util import get_fully_qualified_name

# we allow args because LazyDict calls this method via [] and so we can't set
# what the kwarg names are via that interface
@syft_decorator(typechecking=True, prohibit_args=False)
def _is_string_a_serializable_class_name(
    lazy_dict: LazyDict, fully_qualified_name: str
) -> None:

    """This method exists to allow a LazyDict to determine whether an
    object should actually be in its store - aka has the LazyDict been
    lazy and forgotten to add this object thus far.

    In particular, this is the method for the LazyDict within the fully_qualified_name2type
    dictionary - which is used to map fully qualified module names,
    (i.e., 'syft.core.common.UID') to their type object.

    So this method is designed to ask the question, 'Is self_dict an object
    we can serialize?' If it is, we add it to the LazyDict by adding it to
    lazy_dict._dict. If not, we do nothing.

    We determine whether we can serialize the object according to series of
    checks - as outlined below."""

    # lookup the type from the fully qualified name
    # i.e. "syft.core.common.UID" -> <type UID>
    obj_type = index_syft_by_module_name(fully_qualified_name=fully_qualified_name)

    # Check 1: If the object is a subclass of Serializable, then we can serialize it
    # add it to the lazy dictionary.
    if issubclass(obj_type, Serializable):
        lazy_dict._dict[fully_qualified_name] = obj_type

    # Check 2: If the object a non-syft object which is wrapped by a serializable
    # syft object? Aka, since we can't make non-syft objects subclass from
    # Serializable, have we created a wrapper around this object which does
    # subclass from serializable. Note that we can find out by seeing if we
    # monkeypatched a .serializable_wrapper attribute onto this non-syft class.
    elif hasattr(obj_type, "serializable_wrapper_type"):

        # this 'wrapper' object is a syft object which subclasses Serializable
        # so that we can put logic into it showing how to serialize and
        # deserialize the non-syft object.
        wrapper_type = obj_type.serializable_wrapper_type

        # just renaming the variable since we know something about this variable now
        # just so the code reads easier (the compile will remove this so it won't
        # affect performance)
        non_syft_object_fully_qualified_name = fully_qualified_name
        wrapper_type_fully_qualified_name = get_fully_qualified_name(wrapper_type)

        # so now we should update the dictionary so that in the future we can
        # quickly find the wrapper type from both the non_syft_object's fully
        # qualified name and the wrapper_type's fully qualified name
        lazy_dict[wrapper_type_fully_qualified_name] = wrapper_type
        lazy_dict[non_syft_object_fully_qualified_name] = wrapper_type

    else:
        raise Exception(f"{fully_qualified_name} is not serializable")


fully_qualified_name2type = LazyDict(update_rule=_is_string_a_serializable_class_name)
