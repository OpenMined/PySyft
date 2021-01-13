# stdlib
from typing import Union as TypeUnion
from typing import Any as TypeAny

# syft relative
from ...ast.globals import Globals
from ...ast import add_classes
from ...ast import add_methods
from ...ast import add_modules


# Torch is a dependency for SyferText
PACKAGE_SUPPORT = {"lib": "syfertext", "torch": {"min_version": "1.6.0"}}

def update_ast(ast: TypeUnion[Globals, TypeAny]) -> None:
    syfertext_ast = create_ast()
    ast.add_attr(attr_name="syfertext", attr=syfertext_ast.attrs["syfertext"])

    
def create_ast() -> Globals:

    import syfertext
    from . import spacy_tokenizer    
    
    ast = Globals()


    # Define which SyferText modules to add to the AST
    modules = ['syfertext',
               'syfertext.tokenizers',
               'syfertext.data',
               'syfertext.data.units',
    ]

    # Define which SyferText classes to add to the AST    
    classes = [
        ('syfertext.tokenizers.SpacyTokenizer',
         'syfertext.tokenizers.SpacyTokenizer',
          syfertext.tokenizers.SpacyTokenizer
        ),
        ('syfertext.data.units.TextDoc',
         'syfertext.data.units.TextDoc',
          syfertext.data.units.TextDoc
        ),
    ]

    # Define which methods to add to the AST
    methods = [
        ('syfertext.tokenizers.SpacyTokenizer.__call__',
         'syfertext.data.units.TextDoc'
        ),
        
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
