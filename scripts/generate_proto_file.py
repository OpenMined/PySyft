# stdlib
import os
import re

if __name__ == "__main__":
    try:
        os.remove("./proto/syft_message.proto")
    except OSError:
        pass

    message_pattern = re.compile("message (.*) {")
    package_pattern = re.compile("package (.*);")
    imports = []
    fields = []
    index = 1

    for (root, dirs, files) in os.walk("./proto"):

        obj_path = root[2:]

        for file in files:
            textfile = open(root + "/" + file, "r")
            filetext = textfile.read()
            textfile.close()

            imports.append(f'import "{obj_path + "/" + file}";')
            package = re.findall(package_pattern, filetext)
            if len(package) != 1:
                raise ValueError(f"Can't properly parse {root + file}")
            package_path = package[0]
            matches = re.findall(message_pattern, filetext)
            for match in matches:
                fields.append(f"{package_path}.{match} {match.lower()} = {index};")
                index += 1

    file_imports = "\n".join(imports)
    file_fields = "\n\t\t".join(fields)
    open_marker = "{"
    closed_marker = "}"

    file_template = f"""
syntax = "proto3";
package syft;

{file_imports}

message SyftNative {open_marker}
\toneof data_field {open_marker}
\t\t{file_fields}
\t{closed_marker}
{closed_marker}
"""

    with open("./proto/syft_message.proto", "w") as f:
        f.write(file_template)
