def as_markdown_code(str, lang="python") -> str:
    return f"```{lang}\n{str}\n```"


def as_markdown_python_code(str) -> str:
    return as_markdown_code(str, lang="python")


def markdown_as_class_with_fields(obj, fields, set_defaults=True):
    if set_defaults:
        # add default properties to start of the dict
        fields = {**{"id": obj.id}, **fields}
    _repr_str = f"class {obj.__class__.__name__}:\n  "
    _repr_str += "\n  ".join([f"{k}: {v}" for k, v in fields.items()])
    return as_markdown_python_code(_repr_str)


class CodeMarkdown:
    def __init__(self, code, lang="python"):
        self._code = code
        self._lang = lang

    def _repr_markdown_(self) -> str:
        return as_markdown_code(self._code, self._lang)
