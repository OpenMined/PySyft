# Time Sequence Prediction
This is a toy example for beginners to start with. It is helpful for learning both PyTorch and time sequence prediction. Two LSTM cell units are used in this example to learn some sine wave signals starting at different phases. After learning the sine waves, the network tries to predict the signal values in the future. The results is shown in the picture below.

## Usage

```bash
python generate_sine_wave.py
python train.py [--steps STEPS]

optional arguments:
  --steps     steps to run; default: 15
```

## Result
The initial signal and the predicted results are shown in the image. We first give some initial signals (full line). The network will  subsequently give some predicted results (dash line). It can be concluded that the network can generate new sine waves.
![image](https://cloud.githubusercontent.com/assets/1419566/24184438/e24f5280-0f08-11e7-8f8b-4d972b527a81.png)
