rm -rf build

swig -c++ -python -py3 PyPari.i #1> /dev/null 2> /dev/null
python3 setup.py install #1> /dev/null 2> /dev/null
