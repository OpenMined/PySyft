from __future__ import print_function  # Only Python 2.x

from .client import GridClient

import os
import sys
import subprocess



def run_commands_in(commands, logs, tmp_dir="tmp", cleanup=True, verbose=False):
    os.popen("mkdir " + tmp_dir).read()

    outputs = list()

    cmd = ""
    for i in range(len(commands)):

        if (verbose):
            print(logs[i] + "...")

        cmd = "cd " + str(tmp_dir) + "; " + commands[i] + "; cd ..;"
        o = os.popen(cmd).read()
        outputs.append(str(o))

        if (verbose):
            print("\t" + str(o).replace("\n", "\n\t"))

    if (cleanup):
        os.popen("rm -rf " + tmp_dir).read()

    return outputs


def check_dependency(lib="git", check="usage:", error_msg="Error: please install git.", verbose=False):
    if (verbose):
        sys.stdout.write("\tChecking for " + str(lib) + " dependency...")
    o = os.popen(lib).read()
    if check not in o:
        raise Exception(error_msg)
    if (verbose):
        print("DONE!")

def execute(cmd):
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def launch_on_heroku(grid_name="opengrid5", verbose=True, check_deps=True):
    app_addr = "https://" + str(grid_name) + ".herokuapp.com"
    if (check_deps):
        if (verbose):
            print("Step 0: Checking Dependencies")

        check_dependency(lib="git",
                         check="usage:",
                         error_msg="Missing Git command line dependency - please install it: https://gist.github.com/derhuerst/1b15ff4652a867391f03",
                         verbose=verbose)

        check_dependency(lib="heroku --version",
                         check="heroku/7",
                         error_msg="Missing Heroku command line dependency - please install it: https://toolbelt.heroku.com/",
                         verbose=verbose)

        check_dependency(lib="pip",
                         check="\nUsage:   \n  pip <command> [options]",
                         error_msg="Missing Pip command line dependency - please install it: https://www.makeuseof.com/tag/install-pip-for-python/",
                         verbose=verbose)

        if (verbose):
            sys.stdout.write("\tChecking to see if heroku is logged in...")
        res = os.popen("heroku create app").read()
        if res == 'Enter your Heroku credentials:\n':
            raise Exception("You are not logged in to Heroku. Run 'heroku login'"
                            " from the command line and follow the instructions. "
                            "If you need to create an account. Don't forget to add "
                            " your credit card. Even though you can use Grid on the"
                            " FREE tier, it won't let you activate a Redis database "
                            "without adding your credit card information to your account.")
        if (verbose):
            print("DONE!")

    if (verbose):
        print("\nStep 1: Making sure app name '" + grid_name + "' is available")
    try:
        output = list(execute(("heroku create " + grid_name).split(" ")))
        if (verbose):
            print("\t" + str(output))
    except:
        output = list(execute(("rm -rf tmp").split(" ")))
        print("APP EXISTS: You can already connect to your app at " + app_addr)
        return app_addr

    commands = list()
    logs = list()
    if (verbose):
        print("\nStep 2: Making Sure Redis Database Can Be Spun Up on Heroku (this can take a couple seconds)...")
    try:
        output = list(execute(("heroku addons:create rediscloud -a " + grid_name).split(" ")))
        if (verbose):
            print("\t" + str(output))
    except:

        try:
            print("Cleaning up...")
            output = list(execute(("rm -rf tmp").split(" ")))
            output = list(execute(("heroku destroy " + grid_name + " --confirm " + grid_name).split(" ")))
            print("Success in cleaning up!")
        except:
            print("ERROR: cleaning up... good chance Heroku still has the app or the tmp directory still exists")

        msg = """Creating rediscloud on ⬢ """ + grid_name + """... ⣾
        ⣽⣻⢿⡿⣟⣯⣷⣾⣽Creating rediscloud on ⬢ """ + grid_name + """... !
         ▸    Please verify your account to install this add-on plan (please enter a
         ▸    credit card) For more information, see
         ▸    https://devcenter.heroku.com/categories/billing Verify now at
         ▸    https://heroku.com/verify

         NOTE: OpenMined's Grid nodes can be run on the FREE tier of Heroku,
         but you still have to enter a credit card on Heroku to spin up FREE nodes."""

        raise Exception(msg)

    if (verbose):
        print("\nStep 3: Cleaning up heroku/redis checks...")
    output = list(execute(("heroku destroy " + grid_name + " --confirm " + grid_name).split(" ")))

    commands = list()
    logs = list()

    logs.append("\nStep 4: cleaning up git")
    commands.append("rm -rf .git")

    logs.append("Step 5: cloning heroku app code from Github")
    commands.append("git clone https://github.com/OpenMined/Grid")

    logs.append("Step 6: copying app code from cloned repo")
    commands.append("cp Grid/app/* ./")

    logs.append("Step 7: removing the rest of the cloned code")
    commands.append("rm -rf Grid")

    logs.append("Step 8: Initializing new github (for Heroku)")
    commands.append("git init")

    logs.append("Step 9: Adding files to heroku github")
    commands.append("git add .")

    logs.append("Step 10: Committing files to heroku github")
    commands.append("git commit -am \"init\"")

    run_commands_in(commands, logs, cleanup=False, verbose=verbose)

    logs = list()
    commands = list()

    logs.append("\nStep 11: Pushing code to Heroku (this can take take a few seconds)...")
    commands.append("heroku create " + grid_name)

    logs.append("Step 12: Creating Redis database... (this can take a few seconds)")
    commands.append("heroku addons:create rediscloud -a " + grid_name)

    logs.append("Step 13: Pushing code to Heroku (this can take take a few minutes"
                " - if you're running this in a Jupyter Notebook you can watch progress "
                "in the notebook server terminal)...")
    commands.append("git push heroku master")

    logs.append("Step 14: Cleaning up!")
    commands.append("rm -rf .git")

    run_commands_in(commands, logs, cleanup=True, verbose=verbose)

    print("SUCCESS: You can now connect to your app at " + app_addr)

    return app_addr
