# Debugging PySyft

We currently provide information on how to debug PySyft using Visual Studio Code and PyCharm. If you have any other IDE or debugger that you would like to add to this list, please feel free to contribute.

## VSCode

If you're running Add the following in `.vscode/launch.json`

```
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python Debugger: Remote Attach",
            "type": "debugpy",
            "request": "attach",
            "connect": {
                "host": "localhost",
                "port": 5678
            },
            "justMyCode": false,
            "internalConsoleOptions": "openOnSessionStart",
            "pathMappings": [
                {
                    "localRoot": "${workspaceFolder}/packages/syft/src",
                    "remoteRoot": "/root/app/syft/src"
                },
                {
                    "localRoot": "${workspaceFolder}/packages/grid/backend/grid",
                    "remoteRoot": "/root/app/grid"
                }
            ]
        }
    ]
}
```

Then run

```bash
tox -e dev.k8s.hotreload
```

And you can attach the debugger running on port 5678.

## PyCharm

Add the following to `packages/grid/backend/grid/__init__.py`

```py
import pydevd_pycharm
pydevd_pycharm.settrace('your-local-addr', port=5678, suspend=False)
```

Ensure that `your-local-addr` is reachable from the containers.

Next, replace the debugpy install and `DEBUG_CMD` in `packages/grid/backend/grid/start.sh`:

```bash
# only set by kubernetes to avoid conflict with docker tests
if [[ ${DEBUGGER_ENABLED} == "True" ]];
then
    pip install --user pydevd-pycharm==233.14475.56 # remove debugpy, add pydevd-pycharm
    DEBUG_CMD="" # empty the debug command
fi
```

If it fails to connect, check the backend logs. You might need to install a different pydevd-pycharm version. The version to be installed is shown in the log error message.

Whenever you start a container, it attempts to connect to PyCharm.

```bash
tox -e dev.k8s.hotreload
```
