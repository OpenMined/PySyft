import inspect
import os


def get_caller_file_path() -> str | None:
    stack = inspect.stack()
    print("stack", stack)

    for frame_info in stack:
        if "from syft import test_settings" in str(frame_info.code_context):
            print(f"File: {frame_info.filename}")
            print(f"Line: {frame_info.lineno}")
            print(f"Code: {frame_info.code_context[0].strip()}")
            caller_file_path = os.path.dirname(os.path.abspath(frame_info.filename))
            print("possible path", caller_file_path)
            return caller_file_path

    return None


result = get_caller_file_path()
print(result)
