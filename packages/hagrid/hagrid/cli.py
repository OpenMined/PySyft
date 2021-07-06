import click
from .lib import some_function

@click.group()
def cli():
    pass

@click.command(help="Have your new program say Hi to you!")
@click.argument('name')
def hello(name):

    print(f'Hi, {name}!')
    some_function()


cli.add_command(hello)