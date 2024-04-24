# third party
from IPython.display import HTML
from IPython.display import display

# relative
from ..serde.serializable import serializable
from .syft_object import SYFT_OBJECT_VERSION_2
from .syft_object import SyftObject


@serializable()
class MarkdownDescription(SyftObject):
    # version
    __canonical_name__ = "MarkdownDescription"
    __version__ = SYFT_OBJECT_VERSION_2

    text: str

    def _repr_markdown_(self, wrap_as_python: bool = True, indent: int = 0) -> str:
        style = """
        <style>
            .jp-RenderedHTMLCommon pre {
                background-color: #282c34 !important;
                padding: 10px 10px 10px;
            }
            .jp-RenderedHTMLCommon pre code {
                background-color: #282c34 !important;  /* Set the background color for the text in the code block */
                color: #abb2bf !important;  /* Set text color */
            }
        </style>
        """
        display(HTML(style))
        return self.text
