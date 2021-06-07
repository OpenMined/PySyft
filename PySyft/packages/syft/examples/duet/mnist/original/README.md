# Basic MNIST Example
A PyTorch implementation to train and test on MNIST dataset.

## Usage

### Quick run
```bash
pip install -r requirements.txt
python main.py
# CUDA_VISIBLE_DEVICES=2 python main.py  # to specify GPU id to ex. 2
```

### Details
```bash
usage: python main.py [-h] [--batch-size N] [--test-batch-size tN] [--epochs E] [--lr LR]
               [--gamma M] [--no-cuda] [--dry-run] [--seed S] [--log-interval L]
               [--save-model]

optional arguments:
  -h, --help              show this help message and exit
  --batch-size N          input batch size for training (default: 64)
  --test-batch-size tN     input batch size for testing (default: 1000)
  --epochs E              number of epochs to train (default: 14)
  --lr LR                 learning rate (default: 1.0)
  --gamma M               learning rate step gamma (default: 0.7)
  --seed S                random seed (default: 1)
  --log-interval L        how many batches to wait before logging training status (default: 10)
  --no-cuda               disables CUDA training (default: False)
  --dry-run               quickly check a single pass (default: False)
  --save-model            For Saving the current Model (default: False)
```
