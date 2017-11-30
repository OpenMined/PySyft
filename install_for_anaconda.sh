env_name=${1:-openmined}
conda create -n $env_name python=3.6 numpy jupyter
source activate $env_name
pip install zmq
python setup.py install
