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
import os
import pydevd_pycharm

if os.getenv('DEV_MODE'):
    pydevd_pycharm.settrace('your-local-addr', port=5678, suspend=False)
```

Whenever you start a container in development mode (`DEV_MODE=true`), it attempts to connect to PyCharm. Ensure that `your-local-addr` is reachable from the containers. Then, you can run the following command:

```bash
tox -e dev.k8s.hotreload
```