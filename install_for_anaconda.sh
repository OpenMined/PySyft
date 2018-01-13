env_name=${1:-openmined}
conda create -n $env_name python=3.6 --file requirements.txt
source activate $env_name
python setup.py install
