from typing import Any
from typing import ClassVar
from typing import Type

from ...serde.serializable import serializable
from .action_object import ActionObject
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from .action_object import BASE_PASSTHROUGH_ATTRS
from .action_types import action_types
from typing import Union
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
from transformers.pipelines.text_generation import TextGenerationPipeline
from typing_extensions import Self

class ModelPointer:
    pass

@serializable()
class ModelObject(ActionObject):
    __canonical_name__ = "ModelObject"
    __version__ = SYFT_OBJECT_VERSION_1
    
    syft_internal_type: ClassVar[Type[Any]] = Union[GPT2TokenizerFast, TextGenerationPipeline]
    syft_pointer_type = ModelPointer
    syft_passthrough_attrs = BASE_PASSTHROUGH_ATTRS
    __serde_overrides__ = {"syft_action_data": (lambda x: None, lambda x: None)}
    
    def __post_init__(self) -> None:
        # TODO: load model 
        return ...
    # syft_dont_wrap_attrs = ["dtype"]
    
    # @staticmethod
    # def from_model(model: Any, node) -> Self:
        
        
    #     return ModelObject(syft_action_data=new_model)
        

action_types[GPT2TokenizerFast] = ModelObject
action_types[TextGenerationPipeline] = ModelObject