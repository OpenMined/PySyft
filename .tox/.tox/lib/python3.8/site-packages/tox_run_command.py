import tox.config
from tox import hookimpl


def getargvlist(reader, command):
    return tox.config._ArgvlistReader.getargvlist(reader, command)


@hookimpl
def tox_addoption(parser):
    parser.add_argument('--run-command', help='run this command instead of configured commands')


@hookimpl
def tox_configure(config):
    alternative_cmd = config.option.run_command
    if alternative_cmd:
        for env in config.envlist:
            reader = config.envconfigs[env]._reader
            env_commands = getargvlist(reader, alternative_cmd)
            config.envconfigs[env].commands = env_commands
