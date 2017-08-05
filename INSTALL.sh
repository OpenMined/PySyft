cd syft/he/yashe
rm -rf build
swig -c++ -python -py3 example.i
cd ../../../

python setup.py install
