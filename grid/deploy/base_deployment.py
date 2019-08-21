import sys
import subprocess
import os
from abc import ABC

from grid import utils as gr_utils


class BaseDeployment(ABC):
    """ Abstract Class used to instantiate generic attributes for other deployment classes. """

    def __init__(self, env_vars, verbose: bool = True):
        """
        Args:
            env_vars: Environment vars used to configure component
            verbose: Used to define level of verbosity
            logs: List of logs used in deploy process
            commands: List of commands used in deploy process
        """
        self.env_vars = env_vars
        self.verbose = verbose
        self.logs = list()
        self.commands = list()

    def deploy(self):
        """ Method used to deploy component."""
        raise NotImplementedError("Deployment not specified!")

    def _check_dependency(
        self,
        lib="git",
        check="usage:",
        error_msg="Error: please install git.",
        verbose=False,
    ):
        """ This method checks if the environment has a specific dependency.
            Args:
                dependency_lib : Libs that will be verified.
                check: Specific string to check if app was installed.
                error_msg: If not installed, raise an Exception with this.
                verbose: Used to define level of verbosity.
            Raises:
                RuntimeError: If not installed, raise a RuntimeError Exception.
        """
        if verbose:
            sys.stdout.write("\tChecking for " + str(lib) + " dependency...")
        output = gr_utils.execute_command(lib)
        if check not in output:
            raise RuntimeError(error_msg)
        if verbose:
            print("DONE!")

    def _execute(self, cmd):
        """ Execute a specific bash command and return the result.
            Args:
                cmd: Specific bash command.
            Raises:
                subprocess_exception: Raises an specific subprocess exception
        """
        popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
        for stdout_line in iter(popen.stdout.readline, ""):
            yield stdout_line
        popen.stdout.close()
        return_code = popen.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, cmd)

    def _run_commands_in(
        self, commands, logs, tmp_dir="tmp", cleanup=True, verbose=False
    ):
        """ Run sequentially all commands and logs stored in our list of commands/logs.
            Args:
                commands: List of commands.
                logs: List of logs.
                tmp_dir: Directory used execute these commands.
                cleanup: Flag to choose if tmp_dir will be maintained.
                verbose: Used to define level of verbosity.
            Returns:
                outputs: Output message for each command.
        """
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
