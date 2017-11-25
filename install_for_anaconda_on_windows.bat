call conda create -n %1 python
call activate %1
call pip install -r requirements.txt
call python setup.py install
call python -m ipykernel install --user
