import hashlib

from pathlib import Path
import os
import shutil


def calculate_file_hash(file_path, hash_func=hashlib.sha256):
    """Calculate the hash of a file."""
    hash_obj = hash_func()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def ensure_folder(files, destination_folder):
    """Ensure that specified files are in the destination folder with the same
    hashes. If the destination folder doesn't exist, create it.
    Copy files if missing or hashes differ."""

    # Ensure destination folder exists
    Path(destination_folder).mkdir(parents=True, exist_ok=True)

    for src_file_path in files:
        # Check if the source file exists
        if not os.path.exists(src_file_path):
            print(f"Source file '{src_file_path}' does not exist.")
            continue

        file_name = os.path.basename(src_file_path)
        dest_file_path = os.path.join(destination_folder, file_name)

        # Calculate the hash of the source file
        src_hash = calculate_file_hash(src_file_path)

        # Check if destination file exists and has the same hash
        if os.path.exists(dest_file_path):
            dest_hash = calculate_file_hash(dest_file_path)
            if src_hash == dest_hash:
                print(f"File '{file_name}' is up-to-date.")
                continue  # Skip copying as the file is the same

        # Copy file from source to destination
        shutil.copy2(src_file_path, dest_file_path)
        print(f"Copied '{file_name}' to '{dest_file_path}'.")
