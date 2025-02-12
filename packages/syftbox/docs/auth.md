# Authentication

When using SyftBox for the first time users will be asked to fill in their email, which will receive a registration token. The registration token can be pasted into the terminal, which will result in an access token that will be stored in `<my_syftbox_path>/config.json` and will be used when loggin in.

## Password reset

If users lose their config.json, they can regain access to their account by going through the registration flow again. Users will receive a new email and will be asked to copy a new registration token into the terminal.

## Dev

When you launch a syftbox caching server for development with `just run-server`, by default it will start without authentication. During registration, the client receives a response that indicates that auth is turned off and during login the client just passes a base64 encoded json of your email address. The server will skip any jwt validation.
