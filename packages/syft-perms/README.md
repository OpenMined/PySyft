# syft-perms

User-facing permission API for Syft datasites.

## Dev Setup

```bash
uv pip install -e .
```

## Quick Start

```python
import syft_perms as sp

# 1. Opening files and folders
file = sp.open("data.csv")                    # Open a file
folder = sp.open("my_project/")                # Open a folder
remote = sp.open("syft://alice@datasite.org/data.csv")  # Remote files

# 2. Granting permissions (each level includes all lower permissions)
file.grant_read_access("bob@company.com")      # Can view
file.grant_create_access("alice@company.com")  # Can view + create new files
file.grant_write_access("team@company.com")    # Can view + create + modify
file.grant_admin_access("admin@company.com")   # Full control

# 3. Revoking permissions
file.revoke_read_access("bob@company.com")     # Remove all access
file.revoke_create_access("alice@company.com") # Remove create (keeps read)
file.revoke_write_access("team@company.com")   # Remove write (keeps read/create)
file.revoke_admin_access("admin@company.com")  # Remove admin privileges

# 4. Checking permissions
if file.has_read_access("bob@company.com"):
    print("Bob can read this file")

if file.has_create_access("alice@company.com"):
    print("Alice can create new files")

if file.has_write_access("team@company.com"):
    print("Team can modify this file")

if file.has_admin_access("admin@company.com"):
    print("Admin has full control")

# 5. Understanding permissions with explain
explanation = file.explain_permissions("bob@company.com")
print(explanation)  # Shows why bob has/doesn't have access

# 6. Working with the Files API
all_items = sp.files_and_folders.all()         # Get all files and folders
files_only = sp.files.all()                    # Get only files
folders_only = sp.folders.all()                # Get only folders

paginated = sp.files.get(limit=10, offset=0)   # Get first 10 files
filtered = sp.files.search(admin="me@datasite.org")  # My admin files
sliced = sp.files[0:5]                          # First 5 files using slice

# 7. Moving files while preserving permissions
new_file = file.move_file_and_its_permissions("archive/data.csv")
```

### Permission Hierarchy

- **Read**: View file contents
- **Create**: Read + create new files in folders
- **Write**: Read + Create + modify existing files
- **Admin**: Read + Create + Write + manage permissions
