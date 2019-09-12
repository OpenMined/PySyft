from __future__ import print_function  # Only Python 2.x

import sys
import subprocess
import os

import syft

from grid.client import GridClient
from grid.websocket_client import WebsocketGridClient
from grid import utils as gr_utils
from grid import deploy
from grid.grid_network import GridNetwork

from grid.utils import connect_all_nodes

__all__ = ["workers", "connect_all_nodes", "syft"]


# ======= Providing a friendly API on top of Syft ===============
def encrypt(self, worker_1, worker_2, crypto_provider):
    """tensor.fix_prec().share()"""
    return self.fix_prec().share(worker_1, worker_2, crypto_provider=crypto_provider)


syft.frameworks.torch.tensors.interpreters.native.TorchTensor.encrypt = encrypt
syft.messaging.plan.Plan.encrypt = encrypt


def request_decryption(self):
    """tensor.get().float_prec()"""
    return self.get().float_prec()


syft.frameworks.torch.tensors.interpreters.native.TorchTensor.request_decryption = (
    request_decryption
)


# =============== Heroku related functions =======================


def run_commands_in(commands, logs, tmp_dir="tmp", cleanup=True, verbose=False):
    assert len(commands) == len(logs)
    gr_utils.execute_command("mkdir " + tmp_dir)

    outputs = list()

    cmd = ""
    for i in range(len(commands)):

        if verbose:
            print(logs[i] + "...")

        cmd = "cd " + str(tmp_dir) + "; " + commands[i] + "; cd ..;"
        output = gr_utils.execute_command(cmd)
        outputs.append(str(output))

        if verbose:
            print("\t" + str(output).replace("\n", "\n\t"))

    if cleanup:
        gr_utils.execute_command("rm -rf " + tmp_dir)

    return outputs


def check_dependency(
    lib="git", check="usage:", error_msg="Error: please install git.", verbose=False
):
    if verbose:
        sys.stdout.write("\tChecking for " + str(lib) + " dependency...")
    output = gr_utils.execute_command(lib)
    if check not in output:
        raise Exception(error_msg)
    if verbose:
        print("DONE!")


def execute(cmd):
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def launch_on_heroku(
    grid_name: str,
    app_type: str = "pg_rest_api",
    verbose=True,
    check_deps=True,
    dev_user: str = "OpenMined",
    branch: str = "dev",
):
    """Launches a node as a heroku application. User needs to be logged in to heroku prior to calling this function.

        Args:
            grid_name (str): The name of the node / Heroku application.
            app_type (str): Type of node being deployed to heroku. Defaults to "pg_rest_api".
            verbose (bool): Specifies logging level. Set true for more logs. Default to True.
            check_deps (bool): Checks before deployment for local git, heroku and pip installations. Defaults to True.
            dev_user (str): Github username of the user/ organization whose Grid repo will be used. Leave undefined to use 'OpenMined' repo.
            branch (str): The default branch to use from the Grid repo of the defined dev_user. Leave undefined to use 'dev' branch.
        Returns:
            str: heroku application address (url)

    """
    app_addr = "https://" + str(grid_name) + ".herokuapp.com"
    if check_deps:
        if verbose:
            print("Step 0: Checking Dependencies")

        check_dependency(
            lib="git",
            check="usage:",
            error_msg="Missing Git command line dependency - please install it: https://gist.github.com/derhuerst/1b15ff4652a867391f03",
            verbose=verbose,
        )

        check_dependency(
            lib="heroku --version",
            check="heroku/7",
            error_msg="Missing Heroku command line dependency - please install it: https://toolbelt.heroku.com/",
            verbose=verbose,
        )

        check_dependency(
            lib="pip",
            check="\nUsage:   \n  pip <command> [options]",
            error_msg="Missing Pip command line dependency - please install it: https://www.makeuseof.com/tag/install-pip-for-python/",
            verbose=verbose,
        )

        if verbose:
            sys.stdout.write("\tChecking to see if heroku is logged in...")
        res = gr_utils.execute_command("heroku create app")
        if res == "Enter your Heroku credentials:\n":
            raise Exception(
                "You are not logged in to Heroku. Run 'heroku login'"
                " from the command line and follow the instructions. "
                "If you need to create an account. Don't forget to add "
                " your credit card. Even though you can use Grid on the"
                " FREE tier, it won't let you activate a Redis database "
                "without adding your credit card information to your account."
            )
        if verbose:
            print("DONE!")

    if verbose:
        print("\nStep 1: Making sure app name '" + grid_name + "' is available")
    try:
        output = list(execute(("heroku create " + grid_name).split(" ")))
        if verbose:
            print("\t" + str(output))
    except:
        if os.name != "nt":
            output = list(execute(("rm -rf tmp").split(" ")))
            if verbose:
                print("\t" + str(output))
        print("APP EXISTS: You can already connect to your app at " + app_addr)
        return app_addr

    commands = list()
    logs = list()
    if verbose:
        print(
            "\nStep 2: Making Sure Postgres  Database Can Be Spun Up on Heroku (this can take a couple seconds)..."
        )
    try:
        output = list(
            #            execute(("heroku addons:create heroku-postgresql:hobby-dev -a " + grid_name).split(" ")),
            execute(
                (
                    "heroku addons:create heroku-postgresql:hobby-dev -a " + grid_name
                ).split(" ")
            )
        )
        if verbose:
            print("\t" + str(output))
    except:

        try:
            print("Cleaning up...")
            output = list(execute(("rm -rf tmp").split(" ")))
            output = list(
                execute(
                    ("heroku destroy " + grid_name + " --confirm " + grid_name).split(
                        " "
                    )
                )
            )
            print("Success in cleaning up!")
        except:
            print(
                "ERROR: cleaning up... good chance Heroku still has the app or the tmp directory still exists"
            )

        msg = (
            """Creating heroku-postgresql:hobby-dev on ⬢ """
            + grid_name
            + """... ⣾
        ⣽⣻⢿⡿⣟⣯⣷⣾⣽Creating heroku-postgresql:hobby-dev on ⬢ """
            + grid_name
            + """... !
         ▸    Please verify your account to install this add-on plan (please enter a
         ▸    credit card) For more information, see
         ▸    https://devcenter.heroku.com/categories/billing Verify now at
         ▸    https://heroku.com/verify

         NOTE: OpenMined's Grid nodes can be run on the FREE tier of Heroku,
         but you still have to enter a credit card on Heroku to spin up FREE nodes."""
        )

        raise Exception(msg)

    if verbose:
        print("\nStep 3: Cleaning up heroku/postgres checks...")
    output = list(
        execute(("heroku destroy " + grid_name + " --confirm " + grid_name).split(" "))
    )

    commands = list()
    logs = list()

    logs.append("\nStep 4: cleaning up git")
    commands.append("rm -rf .git")

    # Using the dev user and branch specified. Fetches, clones and then deploys the branch as a heroku application
    # If no dev user/ branch is defined, then it defaults to OpenMined user and dev branch
    logs.append("Step 5: cloning heroku app code from Github")
    if branch:
        commands.append(
            "git clone -b {} https://github.com/{}/Grid".format(branch, dev_user)
        )
    else:
        commands.append("git clone -b dev https://github.com/{}/Grid".format(dev_user))
        logs.append("Checking out dev version...")
        commands.append("git checkout origin/dev")

    logs.append("Step 6: copying app code from cloned repo")
    commands.append("cp -r Grid/app/{}/* ./".format(app_type))

    logs.append("Step 7: removing the rest of the cloned code")
    commands.append("rm -rf Grid")

    logs.append("Step 8: Initializing new github (for Heroku)")
    commands.append("git init")

    logs.append("Step 9: Adding files to heroku github")
    commands.append("git add .")

    logs.append("Step 10: Committing files to heroku github")
    commands.append('git commit -am "init"')

    run_commands_in(commands, logs, cleanup=False, verbose=verbose)

    logs = list()
    commands = list()

    logs.append(
        "\nStep 11: Pushing code to Heroku (this can take take a few seconds)..."
    )
    commands.append("heroku create " + grid_name)

    logs.append("Step 12: Creating Postgres database... (this can take a few seconds)")
    commands.append(f"heroku addons:create heroku-postgresql:hobby-dev -a {grid_name}")

    logs.append(
        "Step 13: Pushing code to Heroku (this can take take a few minutes"
        " - if you're running this in a Jupyter Notebook you can watch progress "
        "in the notebook server terminal)..."
    )
    commands.append("git push heroku master")

    logs.append("Step 14: Create Database")
    commands.append(f"heroku run -a {grid_name} flask db upgrade")

    logs.append("Step 15: Cleaning up!")
    commands.append("rm -rf .git")

    run_commands_in(commands, logs, cleanup=True, verbose=verbose)

    print("SUCCESS: You can now connect to your app at " + app_addr)

    return app_addr
