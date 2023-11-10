# stdlib
import argparse
from pathlib import Path
import re


def convert_github_markdown_to_pypi(input_file, output_file, repo, version):
    with open(input_file, encoding="utf-8") as file:
        content = file.read()

    # Regular expression pattern to match <picture> and <source> tags
    pattern = r"<picture>\s*<source[^>]*>\s*(<img[^>]*>)\s*</picture>"

    # Replace matched patterns with the contents of the <img> tag
    content = re.sub(pattern, r"\1", content)

    # Regular expression patterns to match <img> and <a> tags with relative links
    img_pattern = r'(<img[^>]* src=")(?!https://)([^"]*)"'
    a_pattern = r'(<a href=")(?!https://)([^"]*)"'

    # Replace matched patterns with absolute links
    content = re.sub(
        img_pattern,
        r"\1https://raw.githubusercontent.com/" + repo + "/" + version + r'/\2"',
        content,
    )
    content = re.sub(
        a_pattern,
        r"\1https://github.com/" + repo + "/tree/" + version + r'/\2"',
        content,
    )

    # Write the modified content to the output file
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file",
        type=Path,
        default=None,
        required=True,
        help="Input file path of README file to modify",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        required=True,
        help="Output file path of README file to modify",
    )

    parser.add_argument(
        "--repo",
        type=str,
        default="OpenMined/PySyft",
        help="Repository to use for the PYPI readme file",
    )

    parser.add_argument(
        "--version",
        type=str,
        default=None,
        required=True,
        help="Version to use for the readme file",
    )

    args = parser.parse_args()

    print(">> Args", args.__dict__)
    print(">> Input File:", args.input_file)
    print(">> Output File:", args.output_file)
    print(">> Repo URL:", args.repo)
    print(">> Version:", args.version)

    # Converting the readme file to pypi format
    convert_github_markdown_to_pypi(
        args.input_file, args.output_file, args.repo, args.version
    )

    print("\n\n")
    print("-" * 50)
    print(">> Done")
    print("-" * 50)


if __name__ == "__main__":
    main()
