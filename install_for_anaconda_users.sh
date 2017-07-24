conda create -n openmined python=3.4
source activate openmined
conda install -c anaconda mpc
conda install -c anaconda gmp
conda install -c anaconda mpfr
conda install gmpy2 --channel conda-forge
pip install -r requirements.txt
python setup.py install
