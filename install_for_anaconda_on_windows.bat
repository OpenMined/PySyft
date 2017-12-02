call conda create -n %1 python=3.6 numpy jupyter
call activate %1
call pip install zmq
call python setup.py install
call python -m ipykernel install --user
