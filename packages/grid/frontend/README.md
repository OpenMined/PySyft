![PyGrid logo](https://raw.githubusercontent.com/OpenMined/design-assets/master/logos/PyGrid/horizontal-primary-trans.png)

### Deprecation note

For convenience reasons, this repository is moving into PySyft:

```
https://github.com/OpenMined/PySyft/tree/dev/packages/grid
```

New developments and issues will be managed in this monorepo structure.

# PyGrid Admin

PyGrid Admin is the user interface that allows data owners to manage their
**deployed** PyGrid Domain or Network. This project provides a way to visualize
privacy budgets, configure many infrastucture settings for your Domain or
Network and manage users and datasets.

## Screenshots

<div>
  <img alt="Login" width="30%" src="./docs/screenshots/login.png" />
  <img alt="Users" width="30%" src="./docs/screenshots/users.png" />
  <img alt="Datasets" width="30%" src="./docs/screenshots/datasets.png" />
  <img alt="Tensors" width="30%" src="./docs/screenshots/tensors.png" />
  <img alt="Requests" width="30%" src="./docs/screenshots/requests.png" />
  <img alt="Dashboard" width="30%" src="./docs/screenshots/dashboard.png" />
</diV>

## Deployment

First, add the PyGrid Domain url as an environment variable. Open
`.env.production` and set your PyGrid Domain IP

```bash
NEXT_PUBLIC_API_URL=<PyGrid Domain url>
```

The Admin UI can be hosted in static file services such as Amazon S3, Netlify,
Google Cloud, Vercel or [Heroku](/docs/heroku.md), and can be easily deployed to
your own network. It was deliberately built separately from PyGrid to allow for
the a very simple deployment.

```bash
yarn install
yarn build
yarn next export
```

## Development

Yes! We want your help. Head over to [issues](/../../issues) and help us improve
the look and feel of the project or the provide the functionality that seems to
be missing.

Run the development server:

```bash
git clone git@github.com:OpenMined/PyGrid-Admin.git
cd pygrid-admin
yarn install
yarn dev
```

The PyGrid Admin should be available at
[http://localhost:3000](http://localhost:3000).

## Disclaimer

We are reaching full support for the latest PyGrid Domain API version. And we
know that the interface is not ideal. Please report to us any issues you have
and keep in mind that we are working diligently to provide a better support in
the upcoming versions.

As of now, PyGrid Admin supports the following operations via PyGrid API

| Feature      | READ | CREATE | UPDATE | DELETE |
| ------------ | :--: | :----: | :----: | :----: |
| Users        |  ✅  |   ✅   |   ✅   |   ✅   |
| Groups       |  ✅  |   ✅   |   ✅   |   ✅   |
| Roles        |  ✅  |   ✅   |   ✅   |   ✅   |
| Datasets     |  ✅  |   ✅   |   ❌   |   ✅   |
| Requests     |  ✅  |   -    |   -    |   ✅   |
| Tensors      |  ✅  |   -    |   -    |   ❌   |
| Workers      |  ❌  |   -    |   -    |   ❌   |
| Associations |  ❌  |   -    |   -    |   ❌   |
| Settings     |  ✅  |   -    |   ❌   |   -    |

## Contributing & Support

If you're interested in contributing, please join the **#support** Slack
channel. [Click here to join our Slack community!](https://slack.openmined.org)

## License

[Apache License 2.0](https://github.com/OpenMined/PySyft/blob/master/LICENSE)
[![FOSSA Status](https://app.fossa.io/api/projects/git%2Bgithub.com%2FOpenMined%2FPySyft.svg?type=large)](https://app.fossa.io/projects/git%2Bgithub.com%2FOpenMined%2FPySyft?ref=badge_large)
