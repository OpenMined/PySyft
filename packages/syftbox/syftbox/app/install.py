import json
import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from tempfile import mkdtemp
from types import SimpleNamespace

from typing_extensions import Any, Optional

from syftbox.lib.types import PathLike


def is_git_installed() -> bool:
    """
    Checks if Git is installed on the system.

    Returns:
        bool: `True` if Git is installed, `False` otherwise.

    Functionality:
        - Runs the `git --version` command to check if Git is installed.
        - If the command runs successfully, returns `True`.
        - If the command fails (e.g., Git is not installed), returns `False`.

    Example:
        ```python
        if is_git_installed():
            print("Git is installed on this system.")
        else:
            print("Git is not installed. Please install Git to proceed.")
        ```
        This will print a message indicating whether Git is installed or not.
    """
    try:
        subprocess.run(
            ["git", "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def sanitize_git_path(path: str) -> str:
    """
    Validates and sanitizes a Git repository path, ensuring it matches the required format.

    Args:
        path (str): The Git repository path to validate.

    Returns:
        str: The sanitized Git repository path if it matches the valid pattern.

    Raises:
        ValueError: If the provided path does not match the expected format for a Git repository.

    Functionality:
        - Uses a regular expression pattern to ensure that the given path follows the format `owner/repository`.
        - If the path matches the pattern, returns it as a valid Git path.
        - If the path does not match the pattern, raises a `ValueError` with a descriptive message.

    Example:
        Suppose you have a GitHub path like `OpenMined/logged_in` and want to validate it:
        ```python
        try:
            sanitized_path = sanitize_git_path("OpenMined/logged_in")
        except ValueError as e:
            print(e)
        ```
        If the path is valid, `sanitized_path` will contain the validated GitHub path. If it is not valid, the error message
        "Invalid Git repository path format. (eg: OpenMined/logged_in)" will be printed.
    """

    if path.startswith("http://"):
        path = path.replace("http://", "")

    if path.startswith("https://"):
        path = path.replace("https://", "")

    if path.startswith("github.com/"):
        path = path.replace("github.com/", "")

    # Define a regex pattern for a valid GitHub path
    pattern = r"^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$"

    # Check if the path matches the pattern
    if re.match(pattern, path):
        return path
    else:
        raise ValueError("Invalid Git repository path format. (eg: OpenMined/logged_in)")


def delete_folder_if_exists(folder_path: PathLike) -> None:
    """
    Deletes a folder if it exists at the specified path.

    Args:
        folder_path (PathLike): The path to the folder to be deleted.

    Returns:
        None: This function does not return any value.

    Functionality:
        - Checks if the folder exists at the given path.
        - If the folder exists and is a directory, deletes it and all of its contents using `shutil.rmtree()`.

    Example:
        Suppose you want to delete a folder located at `/tmp/old_clone` if it exists:
        ```python
        delete_folder_if_exists("/tmp/old_clone")
        ```
        This will delete the folder and all of its contents if it exists.
    """
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        shutil.rmtree(folder_path)


def is_repo_accessible(repo_url: str) -> bool:
    """
    Checks if the specified Git repository is accessible.

    Args:
        repo_url (str): The URL of the Git repository to check.

    Returns:
        bool: `True` if the repository is accessible, `False` otherwise.

    Functionality:
        - Uses the `git ls-remote` command to check if the Git repository is accessible.
        - If the command succeeds, returns `True`.
        - If the command fails or times out, returns `False`.

    Example:
        Suppose you want to check if a repository located at `https://github.com/example/repo.git` is accessible.
        You can call the function like this:
        ```python
        is_accessible = is_repo_accessible("https://github.com/example/repo.git")
        ```
        This will return `True` if the repository is accessible, or `False` if it is not.
    """
    try:
        env = os.environ.copy()
        env["GIT_TERMINAL_PROMPT"] = "0"
        subprocess.run(
            ["git", "ls-remote", repo_url],
            check=True,
            env=env,
            capture_output=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def clone_repository(sanitized_git_path: str, branch: str) -> PathLike:
    """
    Clones a Git repository from GitHub to a temporary directory.

    Args:
        sanitized_git_path (str): The Git repository path in the format `owner/repository`.

    Returns:
        str: The path to the cloned repository.

    Raises:
        Exception: If Git is not installed on the system.
        ValueError: If the provided repository path is not accessible.
        CalledProcessError: If there is an error during the cloning process.

    Functionality:
        - Checks if Git is installed on the system by calling `is_git_installed()`.
        - Forms the GitHub repository URL from the provided `sanitized_git_path`.
        - Checks if the repository is accessible by calling `is_repo_accessible()`.
        - Clones the repository to a temporary directory (`/tmp`).
        - Deletes any existing folder in `/tmp` with the same name before cloning.
        - If cloning is successful, returns the path to the cloned repository.
        - If any error occurs during cloning, raises the corresponding exception.

    Example:
        Suppose you want to clone a repository located at `OpenMined/PySyft` to a temporary directory.
        You can call the function like this:
        ```python
        try:
            clone_path = clone_repository("OpenMined/PySyft")
            print(f"Repository cloned to: {clone_path}")
        except Exception as e:
            print(e)
        ```
        This will clone the repository to `/tmp/PySyft` if successful, or print an error message if any issues occur.
    """
    if not is_git_installed():
        raise Exception(
            "git cli isn't installed. Please, follow the instructions"
            + " to install git according to your OS. (eg. brew install git)"
        )
    repo_url = f"https://github.com/{sanitized_git_path}.git"
    if not is_repo_accessible(repo_url):
        raise ValueError(f"Cannot access repository {repo_url}")

    # Clone repository in /tmp
    tmp_path = mkdtemp(prefix="syftbox_app_")
    temp_clone_path = Path(tmp_path, sanitized_git_path.split("/")[-1])

    # Delete if there's already an existent repository folder in /tmp path.
    delete_folder_if_exists(temp_clone_path)

    try:
        subprocess.run(
            [
                "git",
                "clone",
                "-b",
                branch,
                "--single-branch",
                repo_url,
                temp_clone_path,
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        return temp_clone_path
    except subprocess.CalledProcessError as e:
        raise RuntimeError(e.stderr)


def dict_to_namespace(data: Any) -> Any:
    """
    Converts a dictionary (or nested dictionary) to a SimpleNamespace object.

    Args:
        data (dict or list): The data to convert. Can be a dictionary, list of dictionaries, or other types.

    Returns:
        SimpleNamespace or list: A SimpleNamespace object representing the dictionary data,
                                or a list of SimpleNamespace objects if the input is a list.
                                If the input is not a dictionary or list, returns the input as-is.

    Functionality:
        - Recursively converts dictionaries to SimpleNamespace objects.
        - If the data is a list, each item in the list is recursively converted.
        - If the data is neither a dictionary nor a list, returns the data unchanged.

    Example:
        Suppose you have a dictionary with nested data:
        ```python
        data = {
            "user": {
                "name": "Alice",
                "age": 30,
                "address": {
                    "city": "Wonderland",
                    "zipcode": "12345"
                }
            },
            "active": True
        }
        namespace_data = dict_to_namespace(data)
        print(namespace_data.user.name)  # Output: Alice
        print(namespace_data.user.address.city)  # Output: Wonderland
        ```
        This will allow you to access dictionary values using dot notation like attributes.
    """
    if isinstance(data, dict):
        return SimpleNamespace(**{key: dict_to_namespace(value) for key, value in data.items()})
    elif isinstance(data, list):
        return [dict_to_namespace(item) for item in data]
    else:
        return data


def load_config(path: PathLike) -> SimpleNamespace:
    """
    Loads a JSON configuration file and converts it to a SimpleNamespace object.

    Args:
        path (str): The file path to the JSON configuration file.

    Returns:
        SimpleNamespace: A SimpleNamespace object representing the configuration data.

    Raises:
        ValueError: If the file does not exist, is not in JSON format, or does not contain a dictionary.

    Functionality:
        - Checks if the provided file path exists. If not, raises a `ValueError` indicating the file is not found.
        - Opens and reads the JSON file. If the file cannot be decoded or does not contain a dictionary, raises a `ValueError`.
        - Converts the loaded dictionary to a SimpleNamespace object for easy attribute-based access.

    Example:
        Suppose you have a JSON configuration file at `/path/to/config.json` with the following content:
        ```json
        {
            "version": "0.1.0",
            "app": {
                "version": "1.0"
                "env": {
                    "TEST_ENV": "testing",
                },
            },
        }
        ```
        You can load the configuration and access its fields using dot notation:
        ```python
        try:
            config = load_config("/path/to/config.json")
            print(config.app.version)  # Output: MyApp
            print(config.app.env.TEST_ENV)  # Output: True
        except ValueError as e:
            print(e)
        ```
        This will load the configuration and allow access to its values using attribute access.
    """
    if not os.path.exists(path):
        raise ValueError(f"config not found - {path}")
    try:
        error_msg = "File isn't in JSON format."
        with open(path, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(error_msg)
    except json.JSONDecodeError:
        raise ValueError(error_msg)
    return dict_to_namespace(data)


def create_symbolic_link(apps_dir: PathLike, sanitized_path: PathLike) -> str:
    """
    Creates a symbolic link from the application directory in the Syftbox directory to the user's sync folder.

    Args:
        apps_dir (Path): The path to the `apps` directory in the Syftbox configuration folder.
        app_path (str): The actual path of the application directory.
        sanitized_path (str): The sanitized Git repository path in the format `owner/repository`.

    Returns:
        None: This function does not return any value.

    Functionality:
        - Constructs the symbolic link path within the user's sync folder (`apps` folder).
        - If a symlink already exists at the target location, deletes it to avoid conflicts.
        - Creates a new symbolic link pointing from the sync folder to the application directory.

    Example:
        Suppose you want to create a symbolic link for an application located at `/home/user/.syftbox/apps/PySyft`:
        ```python
        create_symbolic_link(
            apps_dir=SyftWorkspace.apps, # ex "/home/user/SyftBox/apis",
            sanitized_path="OpenMined/PySyft"
        )
        ```
        This will create a symbolic link at `<sync_folder>/apps/PySyft` pointing to the application directory.
    """
    # TODO: Create a Symlink function
    # - Handles if path doesn't exists.
    target_symlink_path = f"{apps_dir}/{str(sanitized_path).split('/')[-1]}"

    # Create the symlink
    if os.path.exists(target_symlink_path) and os.path.islink(target_symlink_path):
        os.unlink(target_symlink_path)

    if not os.path.exists(target_symlink_path):
        os.symlink(sanitized_path, target_symlink_path)
    else:
        raise Exception(f"Path exists and isn't a symlink: {target_symlink_path}")
    return target_symlink_path


def move_repository_to_syftbox(apps_dir: Path, tmp_clone_path: PathLike, sanitized_path: PathLike) -> str:
    """
    Moves a cloned Git repository to the Syftbox directory.

    Args:
        tmp_clone_path (str): The file path to the temporarily cloned Git repository.
        sanitized_path (str): The sanitized Git repository path in the format `owner/repository`.

    Returns:
        str: The final destination path of the moved repository.

    Functionality:
        - Constructs the destination path within the Syftbox configuration directory (`apps` folder).
        - Deletes any existing folder at the destination path to avoid conflicts.
        - Moves the repository from the temporary clone path to the destination path.
        - Returns the new path of the moved repository.

    Example:
        Suppose you have cloned a repository to a temporary path `/tmp/syftbox` and want to move it to the Syftbox directory:
        ```python
        output_path = move_repository_to_syftbox("/tmp/PySyft", "OpenMined/PySyft")
        print(output_path)  # Output: /path/to/config/apps/PySyft
        ```
        This will move the cloned repository to the Syftbox `apps` directory and return the final destination path.
    """
    output_path = f"{apps_dir}/{str(sanitized_path).split('/')[-1]}"
    delete_folder_if_exists(output_path)
    shutil.move(tmp_clone_path, output_path)
    return output_path


def run_pre_install(app_config: SimpleNamespace, app_path: str) -> None:
    """
    Runs pre-installation commands specified in the application configuration.

    Args:
        app_config (SimpleNamespace): The configuration object for the application, which is expected to have an `app`
                                      attribute with a `pre_install` attribute containing a list of commands to run.
        app_path (string): The file path to the app folder.

    Returns:
        None: This function does not return any value.

    Functionality:
        - Checks if the `pre_install` attribute exists and contains commands in the application configuration.
        - If the `pre_install` attribute is empty or does not exist, the function returns without executing any command.
        - If there are pre-installation commands, runs them using `subprocess.run()`.

    Example:
        Suppose you have an application configuration that specifies a pre-installation command to install dependencies:
        ```python
        app_config = SimpleNamespace(
            app=SimpleNamespace(pre_install=["echo", "Installing dependencies..."])
        )
        run_pre_install(app_config)
        ```
        This will run the specified pre-installation command using `subprocess.run()`.
    """
    if len(getattr(app_config.app, "pre_install", [])) == 0:
        return

    try:
        subprocess.run(
            app_config.app.pre_install,
            cwd=app_path,
            capture_output=True,
            check=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(e.stderr)


def run_post_install(app_config: SimpleNamespace, app_path: str) -> None:
    """
    Runs post-installation commands specified in the application configuration.

    Args:
        app_config (SimpleNamespace): The configuration object for the application, which is expected to have an `app`
                                      attribute with a `post_install` attribute containing a list of commands to run.

    Returns:
        None: This function does not return any value.

    Functionality:
        - Checks if the `post_install` attribute exists and contains commands in the application configuration.
        - If the `post_install` attribute is empty or does not exist, the function returns without executing any command.
        - If there are post-installation commands, runs them using `subprocess.run()`.

    Example:
        Suppose you have an application configuration that specifies a post-installation command to perform cleanup:
        ```python
        app_config = SimpleNamespace(
            app=SimpleNamespace(post_install=["echo", "Performing post-installation cleanup..."])
        )
        run_post_install(app_config)
        ```
        This will run the specified post-installation command using `subprocess.run()`.
    """
    if len(getattr(app_config.app, "post_install", [])) == 0:
        return

    try:
        subprocess.run(
            app_config.app.post_install,
            cwd=app_path,
            capture_output=True,
            check=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(e.stderr)


def check_os_compatibility(app_config: SimpleNamespace) -> None:
    """
    Checks whether the current operating system is compatible with the application based on the configuration.

    Args:
        app_config: The configuration object for the application, which is expected to have an `app` attribute
                    with a `platforms` attribute containing a list of supported operating systems.

    Returns:
        None: This function does not return any value.

    Raises:
        OSError: If the current operating system is not supported by the application.

    Functionality:
        - Uses the `platform.system()` function to determine the current operating system.
        - Checks the application's configuration (`app_config`) for a list of supported operating systems.
        - If no platforms are defined in the configuration, the function simply returns without doing anything.
        - If the current operating system is not in the list of supported platforms, raises an `OSError`.

    Example:
        Suppose you have an application configuration that specifies supported platforms as `['Windows', 'Linux']`.
        The function will determine the current operating system and raise an `OSError` if it is not supported:
        ```python
        try:
            check_os_compatibility(app_config)
        except OSError as e:
            print(e)
        ```
        If the current OS is not in the supported platforms list, the message "Your OS isn't supported by this app." will be printed.
    """
    os_name = platform.system().lower()
    supported_os = getattr(app_config.app, "platforms", [])

    # If there's no platforms field in config.json, just ignore it.
    if len(supported_os) == 0:
        return

    is_compatible = False
    for operational_system in supported_os:
        if operational_system.lower() == os_name:
            is_compatible = True

    if not is_compatible:
        raise OSError("Your OS isn't supported by this app.")


def get_current_commit(app_path: str) -> str:
    """
    Retrieves the current commit hash for a Git repository located at the specified path.

    Args:
        app_path (str): The file path to the Git repository.

    Returns:
        str: The current commit hash of the repository if the command is successful.
             If an error occurs, returns an error message describing the failure.

    Functionality:
        - Uses the `git rev-parse HEAD` command to get the current commit hash.
        - If the command succeeds, returns the commit hash as a string.
        - If the command fails (e.g., if the provided path is not a valid Git repository),
          returns an error message detailing what went wrong.

    Example:
        Suppose you have a Git repository at `/path/to/repo` and want to retrieve its current commit hash.
        You can call the function like this:
        ```python
        commit_hash = get_current_commit("/path/to/repo")
        ```
        This will return the commit hash if the repository exists and the command runs successfully,
        or an error message if there is an issue with the command.
    """
    try:
        # Navigate to the repository path and get the current commit hash
        commit_hash = (
            subprocess.check_output(["git", "-C", app_path, "rev-parse", "HEAD"], stderr=subprocess.STDOUT)
            .strip()
            .decode("utf-8")
        )
        return commit_hash
    except subprocess.CalledProcessError:
        return "local"


def update_app_config_file(app_path: str, sanitized_git_path: str, app_config: SimpleNamespace) -> None:
    """
    Updates the `app.json` configuration file with the current commit and version information of an application.

    Args:
        app_path (str): The file path of the application.
        sanitized_git_path (str): The sanitized path representing the Git repository.
        app_config: The configuration object for the application, which is expected to have an `app` attribute
                    with a `version` attribute, if available.

    Returns:
        None: This function modifies the `app.json` configuration file in place and returns nothing.

    Functionality:
        - Normalizes the provided application path.
        - Determines the configuration directory by navigating two levels up from the application path.
        - Checks if an `app.json` file exists in the configuration directory.
            - If it exists, loads its contents into a dictionary.
            - If it does not exist, creates an empty dictionary for new configuration entries.
        - Retrieves the current commit information of the application using the `get_current_commit` function.
        - If the application version is available from the `app_config` object, includes it in the configuration.
        - Updates the `app.json` configuration file with the new commit and version information under the key
          specified by `sanitized_git_path`.
        - Writes the updated configuration back to the `app.json` file with indentation for readability.

    Example:
        Suppose you have an application located at `/path/to/app` and you want to update the `app.json` file
        with the latest commit and version. You can call the function like this:
        ```python
        update_app_config_file("/path/to/app", "my_sanitized_git_path", app_config)
        ```
        This will update or create entries in `app.json` for the given Git path, storing commit and version details.
    """
    normalized_app_path = os.path.normpath(app_path)

    conf_path = os.path.dirname(os.path.dirname(normalized_app_path))

    app_json_path = conf_path + "/app.json"
    app_json_config = {}

    if os.path.exists(app_json_path):
        # Read from it.
        with open(app_json_path, "r") as app_json_file:
            app_json_config = json.load(app_json_file)

    app_version = None
    if getattr(app_config.app, "version", None) is not None:
        app_version = app_config.app.version

    current_commit = get_current_commit(normalized_app_path)
    if current_commit == "local":
        app_version = "dev"

    app_json_config[sanitized_git_path] = {
        "commit": current_commit,
        "version": app_version,
        "path": app_path,
    }

    with open(app_json_path, "w") as json_file:
        json.dump(app_json_config, json_file, indent=4)


def check_app_config(tmp_clone_path: PathLike) -> Optional[SimpleNamespace]:
    app_config_path = Path(tmp_clone_path) / "config.json"
    if os.path.exists(app_config_path):
        app_config = load_config(app_config_path)
        check_os_compatibility(app_config)
        return app_config
    return None


@dataclass
class InstallResult:
    app_name: str
    app_path: Path
    error: Optional[Exception]
    details: Optional[str]


def install(apps_dir: Path, repository: str, branch: str) -> InstallResult:
    """
    Installs an application by cloning the repository, checking compatibility, and running installation scripts.

    Args:
        apps_dir (Path): Path where app will be installed.

    Returns:
        None: If the installation is successful.
        Tuple[str, Exception]: If an error occurs during any installation step, returns a tuple with the step description and the exception raised.

    Functionality:
        - Parses command-line arguments to get the Git repository to install.
        - Performs a series of steps to install the application, including:
            1. Sanitizing the Git repository path.
            2. Cloning the repository to a temporary directory.
            3. Loading the application's configuration (`config.json`).
            4. Checking platform compatibility.
            5. Moving the repository to the Syftbox directory.
            6. Creating a symbolic link on the user's desktop.
            7. Running pre-installation commands.
            8. Running post-installation commands.
            9. Updating the `apps.json` file to include the installed application.
        - If any step fails, returns the step description and the exception raised.

    Example:
        Suppose you have a client configuration and want to install an application from a repository:
        ```python
        result = install(Path("~/.syftbox/apps"), "OpenMined/PySyft", "main")
        if result.error:
            print(f"Error installing {result.app_name}: {result.error}")
            print(f"Failed at step: {result.details}")
        else:
            print(f"Successfully installed {result.app_name} at {result.app_path}")
        ```
        This will install the application, and if an error occurs, it will indicate the step where the failure happened.
    """
    step = ""
    try:
        # NOTE:
        # Sanitize git repository path
        # Handles: bad format repository path.
        # Returns: Sanitized repository path.
        step = "checking app name"

        sanitized_path = repository
        if not os.path.exists(repository):
            sanitized_path = sanitize_git_path(repository)

            # NOTE:
            # Clones the app repository
            # Handles: Git cli tool not installed.
            # Handles: Repository path doesn't exits / isn't public.
            # Handles: If /tmp/apps/<repository_name> already exists (replaces it)
            # Returns: Path where the repository folder was cloned temporarily.
            step = "pulling App"
            tmp_clone_path = clone_repository(sanitized_path, branch)

            # NOTE:
            # Load config.json
            # Handles: config.json doesn't exist in the pulled repository
            # Handles: config.json version is different from syftbox config version.
            # Returns: Loaded app config as SimpleNamespace instance.
        else:
            tmp_clone_path = os.path.abspath(repository)

        # make optional
        app_config: Optional[SimpleNamespace] = None
        try:
            check_app_config(tmp_clone_path)
        except Exception:
            # this function is run in cli context
            # dont loguru here, either rprint or bubble up the error
            app_config = None

        # NOTE:
        # Moves the repository from /tmp to ~/.syftbox/apps/<repository_name>
        # Handles: If ~/.syftbox/apps/<repository_name> already exists (replaces it)
        if not os.path.exists(repository):
            app_config_path = move_repository_to_syftbox(
                apps_dir,
                tmp_clone_path=tmp_clone_path,
                sanitized_path=sanitized_path,
            )
        else:
            # Creates a Symbolic Link ( ~/Desktop/Syftbox/app/<rep> -> ~/.syftbox/apps/<rep>)
            # Handles: If ~/.syftbox/apps/<repository_name> already exists (replaces it)
            step = "creating Symbolic Link"
            app_config_path = create_symbolic_link(
                apps_dir=apps_dir,
                sanitized_path=tmp_clone_path,
            )

        # NOTE:
        # Executes config.json pre-install command list
        # Handles: Exceptions from pre-install command execution
        if app_config:
            step = "running pre-install commands"
            run_pre_install(app_config, app_config_path)

        # NOTE:
        # Executes config.json post-install command list
        # Handles: Exceptions from post-install command execution
        if app_config:
            step = "running post-install commands"
            run_post_install(app_config, app_config_path)

        # NOTE:
        # Updates the apps.json file
        # Handles: If apps.json file doesn't exist yet.
        # Handles: If apps.json already have the repository_name  app listed.
        # Handles: If apps.json exists but doesn't have the repository_name app listed.
        if app_config:
            step = "updating apps.json config"
            update_app_config_file(app_config_path, sanitized_path, app_config)

        app_dir = Path(app_config_path)
        return InstallResult(app_name=app_dir.name, app_path=app_dir, error=None, details=None)
    except Exception as e:
        return InstallResult(app_name="", app_path=Path(""), error=e, details=step)
