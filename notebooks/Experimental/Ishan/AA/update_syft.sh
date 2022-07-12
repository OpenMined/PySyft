echo "CLoning Pysyft";
git clone -b model_training_tests https://github.com/OpenMined/PySyft.git;
echo "Installing Hagrid/PySyft";
pip install -e PySyft/packages/hagrid;
pip install -e PySyft/packages/syft;
echo "Landing Domain ... ";
hagrid land all;
echo "Launching Domain .. hahaha >:)";
hagrid launch domain;
