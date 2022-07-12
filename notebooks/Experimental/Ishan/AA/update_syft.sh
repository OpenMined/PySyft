hagrid land all;
git clone -b model_training_tests https://github.com/OpenMined/PySyft.git;
pip install -e PySyft/packages/hagrid;
pip install -e PySyft/packages/syft;
hagrid launch domain;
