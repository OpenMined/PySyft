env_name=${1:-openmined}
conda create -n $env_name python=3.6
source activate $env_name
pip install -r requirements.txt
python setup.py install
