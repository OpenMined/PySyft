# stdlib
from typing import Dict
from typing import Union

# third party
from packaging import version
import syfertext

# syft relative
from ...ast.globals import Globals
from ...ast import add_classes
from ...ast import add_methods
from ...ast import add_modules


# SyferText Version
SYFERTEXT_VERSION = version.parse(syfertext.__version__)


def get_return_type(support_dict: Union[str, Dict[str, str]]) -> str:

    if isinstance(support_dict, str):
        return support_dict
    
    else:
        return support_dict['return_type']


def version_supported(support_dict: Union[str, Dict[str, str]]) -> bool:
    
    if isinstance(support_dict, str):
        return True
    
    else:
        
        # if we are on either side of the min or max versions we don't support this operation
        if ("min_version" in support_dict
            and SYFERTEXT_VERSION < version.parse(support_dict["min_version"])
        ):
            return False
        
        if ("max_version" in support_dict
            and SYFERTEXT_VERSION > version.parse(support_dict["max_version"])
        ):
            return False
        
        return True

    
def create_syfertext_ast() -> Globals:

    ast = Globals()


    # Define which SyferText modules to add to the AST
    modules = ['syfertext',
               'syfertext.tokenizers']

    # Define which SyferText classes to add to the AST    
    classes = [
        ('syfertext.tokenizers.DefaultTokenizer', 'syfertext.tokenizers.DefaultTokenizer', syfertext.tokenizers.DefaultTokenizer)
    ]

    # Define which methods to add to the AST
    methods = [
        ('syfertext.tokenizers.DefaultTokenizer.__call__', 'syft.lib.python.List')
    ]


    add_modules(ast, modules)
    add_classes(ast, classes)
    add_methods(ast, methods)    
    
    
    for klass in ast.classes:
        
        klass.create_pointer_class()
        klass.create_send_method()
        klass.create_serialization_methods()
        klass.create_storable_object_attr_convenience_methods()
        
    return ast    
