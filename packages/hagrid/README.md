# hagrid

Use this cli to deploy PyGrid Domain and Network nodes on your local machine.

A Hagrid is a HAppy GRID!

## Installation
```bash
pip install -r requirements.txt
pip install -e .
```

> **_NOTE:_**  Don't use `python setup.py install` or `pip install .` without the `-e`

## Launch a Node

![alt text](cli2.png)

## A Few Example Commands

Start a node with:
```bash
hagrid launch slytherin
```
... and then stop it with:
```bash
hagrid land slytherin
```
You can specify ports if you want to:
```bash
hagrid launch Hufflepuff House --port 8081
```
... but if you don't it'll find an open one for you
```bash
// finds hufflepuff already has 8081... tries 8082
hagrid launch ravenclaw
```
You can also specify the node type (domain by default)
```bash
hagrid launch gryffendor --type network
```
## Credits

**Super Cool Code Images** by [Carbon](https://carbon.now.sh/)
