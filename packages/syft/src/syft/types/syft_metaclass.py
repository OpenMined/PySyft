from typing import Any
from typing import DefaultDict
from typing import Generator
from typing import OrderedDict
from typing import Tuple
from typing import Type

# third party
from pydantic.fields import ModelField
from pydantic.main import BaseModel
from pydantic.main import ModelMetaclass

# relative
from ..serde.recursive_primitives import recursive_serde_register_type
from ..serde.serializable import serializable


TupleGenerator = Generator[Tuple[str, Any], None, None]


@serializable()
class Empty:
    pass


class PartialModelMetaclass(ModelMetaclass):
    def __new__(
        meta: Type["PartialModelMetaclass"], *args: Any, **kwargs: Any
    ) -> "PartialModelMetaclass":
        cls = super().__new__(meta, *args, **kwargs)
        cls_init = cls.__init__

        def __init__(self: BaseModel, *args: Any, **kwargs: Any) -> None:
            fields_map: DefaultDict[ModelField, Tuple[Any, bool]] = DefaultDict(lambda: None)

            def optionalize(fields: Dict[str, ModelField]):
                for _, field in fields.items():
                    if field.required is not None:
                        fields_map[field] = (field.type_, field.required)
                        # If field has None allowed as a value
                        # then it becomes a required field.
                        if field.allow_none and field.name in kwargs:
                            field.required = True
                        else:
                            field.required = False
                        if inspect.isclass(field.type_) and issubclass(
                            field.type_, BaseModel
                        ):
                            field.populate_validators()
                            if field.sub_fields is not None:
                                for sub_field in field.sub_fields:
                                    sub_field.type_ = field.type_
                                    sub_field.populate_validators()
                                optionalize(field.type_.__fields__)

            # Make fields and fields of nested model types optional
            optionalize(cls.__fields__)

            # Transform kwargs that are PartialModels to their dict() forms. This
            # will exclude `None` (see below) from the dictionary used to construct
            # the temporarily-partial model field, avoiding ValidationErrors of
            # type type_error.none.not_allowed.
            kwargs2 = OrderedDict()
            for kwarg, value in kwargs.items():
                if isinstance(value, PartialModel):
                    kwargs2[kwarg] = value.dict()
                elif isinstance(value, (tuple, list)):
                    kwargs2[kwarg] = type(value)(
                        v.dict() if isinstance(v, PartialModel) else v for v in value
                    )

            # Validation is performed in __init__, for which all fields are now optional
            cls_init(self, *args, **kwargs2)

            # Restore requiredness
            for field, (type_, required) in fields_map.items():
                field.type_, field.required = type_, required

        cls.__init__ = __init__

        def iter_exclude_empty(self) -> TupleGenerator:
            for key, value in self.__dict__.items():
                if value is not Empty:
                    yield key, value

        cls.__iter__ = iter_exclude_empty

        return cls


recursive_serde_register_type(PartialModelMetaclass)
