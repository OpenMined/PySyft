# Syft Duet PyTorch Examples
The following examples were taken from the official [PyTorch examples](https://github.com/pytorch/examples) repo.

## Data Owner and Data Scientist
Each example is split into two notebooks, one for the data owner who wishes to protect their valuable and private training data and one for the data scientist who has some problem they wish to solve and a test set they can use to determine if the data owner's data will help. Together they are able to collaborate over two notebooks to construct and train models, evaluate and share metrics upon request and do inference on test or individual data items.

## Examples

- [Image classification (MNIST) using Convnets](./mnist)
    - [README.md](./mnist/README.md)
    - [Data Owner Notebook](./mnist/MNIST_Syft_Data_Owner.ipynb)
    - [Data Scientist Notebook](./mnist/MNIST_Syft_Data_Scientist.ipynb)

- [Word level Language Modeling using LSTM RNNs](./word_language_model)
    - [README.md](./word_language_model/README.md)
    - [Data Owner Notebook](./word_language_model/Wikitext_Syft_Data_Owner.ipynb)
    - [Data Scientist Notebook](./word_language_model/Wikitext_Syft_Data_Scientist.ipynb)

- [Implement the Neural Style Transfer algorithm on images](./fast_neural_style)
    - [README.md](./fast_neural_style/README.md)
    - [Data Owner Notebook](./fast_neural_style/StyleTransfer_Syft_Data_Owner.ipynb)
    - [Data Scientist Notebook](./fast_neural_style/StyleTransfer_Syft_Data_Scientist.ipynb)

- [Variational Auto-Encoders](./vae)
    - [README.md](./vae/README.md)
    - [Data Owner Notebook](./vae/AutoEncoder_Syft_Data_Owner.ipynb)
    - [Data Scientist Notebook](./vae/AutoEncoder_Syft_Data_Scientist.ipynb)

- [Training a CartPole to balance in OpenAI Gym with actor-critic](./reinforcement_learning)
    - [README.md](./reinforcement_learning/README.md)
    - [Data Owner Notebook](./reinforcement_learning/Reinforcement_Syft_Data_Owner.ipynb)
    - [Data Scientist Notebook](./reinforcement_learning/Reinforcement_Syft_Data_Scientist.ipynb)

- [Time sequence prediction - use an LSTM to learn Sine waves](./time_sequence_prediction)
    - [README.md](./time_sequence_prediction/README.md)
    - [Data Owner Notebook](./time_sequence_prediction/TimeSequence_Syft_Data_Owner.ipynb)
    - [Data Scientist Notebook](./time_sequence_prediction/TimeSequence_Syft_Data_Scientist.ipynb)

- [Superresolution using an efficient sub-pixel convolutional neural network](./super_resolution)
    - [README.md](./super_resolution/README.md)
    - [Data Owner Notebook](./super_resolution/SuperResolution_Syft_Data_Owner.ipynb)
    - [Data Scientist Notebook](./super_resolution/SuperResolution_Syft_Data_Scientist.ipynb)

- [Natural Language Inference (SNLI) with GloVe vectors, LSTMs, and torchtext](./snli)
    - [README.md](./snli/README.md)
    - [Data Owner Notebook](./snli/SNLI_Syft_Data_Owner.ipynb)
    - [Data Scientist Notebook](./snli/SNLI_Syft_Data_Scientist.ipynb)

- [Generative Adversarial Networks (DCGAN)](./dcgan)
    - [README.md](./dcgan/README.md)
    - [Data Owner Notebook](./dcgan/DCGAN_Syft_Data_Owner.ipynb)
    - [Data Scientist Notebook](./dcgan/DCGAN_Syft_Data_Scientist.ipynb)
