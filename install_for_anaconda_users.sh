conda create -n $1 python
source activate $1
conda install -c conda-forge gmpy2
pip install -r requirements.txt
python setup.py ${2-install}
python -m ipykernel install --user