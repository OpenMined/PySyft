call conda create -n %1 python=3.6 --file requirements.txt
call activate %1
call python setup.py install
call python -m ipykernel install --user
