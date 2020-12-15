# Reinforcement Learning Training Example
A reinforcement learning implementation with [PyTorch](https://github.com/pytorch/pytorch) and [gym](https://github.com/openai/gym)

## Usage

```bash
pip install -r requirements.txt
# For REINFORCE
python reinforce.py
# For actor critic:
python actor_critic.py
```

## Details

### For reinforce:
```bash
usage: python reinforce.py [--gamma G] [--seed S] [--render] [--log-interval N]

  --gamma G           discount factor (default: 0.99)
  --seed S            random seed (default: 543)
  --render            renders the environment
  --log-interval N    interval between training status logs (default: 10)
```
### For actor_critic:
```bash
usage: python actor_critic.py [--gamma G] [--seed S] [--render] [--log-interval N]

  --gamma G           discount factor (default: 0.99)
  --seed S            random seed (default: 543)
  --render            renders the environment
  --log-interval N    interval between training status logs (default: 10)
```
