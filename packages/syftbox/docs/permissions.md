# Permissions

Users can define permissions for paths (files or folders) in their syftbox. Permissions define which other users can read, create, update or delete specific paths. Users can also invite other users to set permissions for specific paths.

## Permission types and syftperm.yaml files

Permissions are defined by creating a set of `syftperm.yaml` files in the filetree of your datasite. A `syftperm.yaml` file can define permissions for all paths lower in the directory structure by defining a set of rules. We have 4 permission `bits`:

- `read`: Can read the file
- `create`: Can create a new file for a particular path
- `write`: Can update an existing file for a particular path
- `admin`: Can change the contents of `syftperm.yaml` files for a particular path

### syftperm.yaml file format

An example of a such a set of rules in a `syftperm.yaml` file is:

```
- permission: read
  path: x.txt
  user: user@example.org

- permission: write
  user: *
  type: disallow
```

### Rule arguments

A rule has the following arguments:

- `permission`: either a single permission, e.g. `read` or a list of permissions, e.g. `["read", "write"]`. Accepted permissions are: `read`, `create`, `write`, `admin`
- `user`: Can be a specific user email, e.g. `user@example.org` or `*`
- `type`: either `allow` or `disallow`. The default is `allow`
- `path`: A path is a unix style glob pattern. You can only use patterns for paths that are in the current directory or lower directory, not parent directories. Currently its not supporting `[]` or `{}` syntax. It also accepts `{useremail}` as a value, which will resolve to a specific user. Valid examples of the `path` values are: `*`: all paths in the current directory `*.txt`: all txt files in the current directory. `**`: all paths recursively. `{useremail}/*.txt`, which will match all txt files for a specific user folder.

## Combining rules

The final set of `read`, `create`, `write`, `admin` permissions is computed by combining the rules. First we sort all the rules by file depth and rule number. The rules are then combined by overriding earlier rules top to bottom as follows:

- By default any datasite owner has all permissions to anything in their datasite. This cannot be overridden. By default, any other user has no permissions to anything in the datasite.
- for any permission in a rule, the permission is added or removed from the final set of permissions for all the users specified by the `user` argument for all the paths specified by the `path` argument. Depending on the `type` argument, the permission is either added or removed.

Permissions "bits" (`read`, `create`, `write`, `admin`) are stored using bitwise independent permissions bits. In general this means that having one permission does not imply that you have the other permissions. There are three exceptions to this:

- if you have admin permissions, you have all other permissions automatically.
- You will only have effective `write` or `create` permissions, if you also have `read` permissions. This is because with syncing, writing to files becomes challenging without `read` permissions.
- any datasite owner can `read`, `create`, `write`, or change any permissions for any path.

## Aliases (future)

Currently not supported: we are planning to add aliases like `creator` which implies `read+create` and `updater` which implies `read+write`.
