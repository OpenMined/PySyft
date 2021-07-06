# hagrid

A Hagrid is a HAppy GRID.

## Requirements

* Python
* setuptoolss
* click

## Development

1. The visit the `hagrid/cli.py` module to add some commands. This template uses the [Click](https://click.palletsprojects.com/en/7.x/) library for simple, beautiful, command line interfaces, by the same guys that made [Flask](https://flask.palletsprojects.com/en/1.1.x/)

    ```python
    @click.command(help="Absolutely obliterate your target")
    @click.argument('target')
    def nuke(target):

        print(f'Launching nukes at {target}!!!.')
        print("Now you'll have to go to Walmart :/")

    cli.add_command(nuke)
    ```

2. Finally, in your terminal or command prompt, run the for development mode (you won't have to reinstal cli tool to reflect new changes as you develop):

    ```bash
    python setup.py develop
    ```

    or for production

    ```bash
    python setup.py install
    ```

3. This automatially creates a symlink to your program. Run and check the output!

    ```bash
    Usage: pycli [OPTIONS] COMMAND [ARGS]...

    Options:
    --help  Show this message and exit.

    Commands:
    hello  Have your new program say Hi to you!
    nuke   Absolutely obliterate your target
    ```

## Conventions

This template aims to reflect python standards and best practices by having a proper module and package oriented structure. But feel free to customize for your own needs!

## Credits

**Super Cool Code Images** by [Carbon](https://carbon.now.sh/)
