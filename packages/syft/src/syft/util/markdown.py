def as_markdown_python_code(str) -> str:
    return f"```python\n{str}\n```"


def markdown_as_class_with_fields(obj, fields, set_defaults=True):
    if set_defaults:
        # add default properties to start of the dict
        fields = {**{"id": obj.id}, **fields}
    _repr_str = f"class {obj.__class__.__name__}:\n  "
    _repr_str += "\n  ".join([f"{k}: {v}" for k, v in fields.items()])
    return as_markdown_python_code(_repr_str)
