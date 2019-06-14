# Pysyft on remote data

Make sure you installed pysyft and torch correctly. I advice you to follow this steps: https://github.com/OpenMined/PySyft/blob/dev/INSTALLATION.md

First, you should run the files startserverworkerhospital1.py and startserverworkerhospital2.py after modifying your IP address accordingly.

Then you can open either:
- PredictPytorchBreastCancerWithoutValidationAndTest.ipynb to test pure PyTorch model
- PredictPytorchBreastCancerFederateWith2workers.ipynb to test the Federal Learning with PySyft. (don't forget to change your IP address accordingly).

