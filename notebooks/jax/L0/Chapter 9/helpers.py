# Code from: https://github.com/RobertTLange/code-and-blog/blob/master/04_jax_intro/helpers.py

# third party
import matplotlib.pyplot as plt
import numpy as onp
import seaborn as sns

sns.set(
    context="poster",
    style="white",
    font="sans-serif",
    font_scale=1,
    color_codes=True,
    rc=None,
)


def plot_mnist_examples(train_loader):
    # Plot some MNIST example samples
    for data, target in train_loader:
        images = onp.array(data[:4, ...]).reshape(4, 28, 28)
        target = onp.array(target[:4, ...])
        break

    fig, axs = plt.subplots(1, 4, figsize=(10, 5))
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(images[i, ...], cmap="Greys")
        ax.set_title("Label: {}".format(target[i]), fontsize=30)
        ax.set_axis_off()
    fig.tight_layout()


def plot_mnist_performance(train_loss, train_acc, test_acc, sup_title="Loss Curve"):
    """Visualize the learning performance of a classifier on MNIST"""
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].plot(train_loss)
    axs[0].set_xlabel("# Batch Updates")
    axs[0].set_ylabel("Batch Loss")
    axs[0].set_title("Training Loss")

    axs[1].plot(train_acc, label="Training")
    axs[1].plot(test_acc, label="Test")
    axs[1].set_xlabel("# Epochs")
    axs[1].set_ylabel("Accuracy")
    axs[1].set_title("Prediction Accuracy")
    axs[1].legend()

    # Give data more room to bloom!
    for i in range(2):
        axs[i].spines["top"].set_visible(False)
        axs[i].spines["right"].set_visible(False)

    fig.suptitle(sup_title, fontsize=25)
    fig.tight_layout(rect=[0, 0.03, 1, 0.925])


def generate_ou_process(batch_size, num_dims, mu, tau, sigma, noise_std, dt=0.1):
    """Ornstein-Uhlenbeck process sequences to train on"""
    ou_x = onp.zeros((batch_size, num_dims))
    ou_x[:, 0] = onp.random.random(batch_size)
    for t in range(1, num_dims):
        dx = -(ou_x[:, t - 1] - mu) / tau * dt + sigma * onp.sqrt(
            2 / tau
        ) * onp.random.normal(0, 1, batch_size) * onp.sqrt(dt)
        ou_x[:, t] = ou_x[:, t - 1] + dx

    ou_x_noise = ou_x + onp.random.multivariate_normal(
        onp.zeros(num_dims), noise_std * onp.eye(num_dims), batch_size
    )

    return ou_x, ou_x_noise


def plot_ou_process(x, x_tilde=None, x_pred=None, title=r"Ornstein-Uhlenbeck Process"):
    """Visualize an example datapoint (OU process or convolved noise)"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(range(len(x)), x, label="Ground Truth", alpha=0.75)
    if x_tilde is not None:
        ax.plot(range(len(x_tilde)), x_tilde, label="Noisy", alpha=0.75)
    if x_pred is not None:
        ax.plot(range(len(x_pred)), x_pred, label="Prediction")
    ax.set_ylabel(r"OU Process")
    ax.set_xlabel(r"Time $t$")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=12)
    return


def plot_ou_loss(train_loss, title="Train Loss - OU GRU-RNN"):
    """Visualize the learning performance of the OU process RNN"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(train_loss)
    ax.set_xlabel("# Batch Updates")
    ax.set_ylabel("Batch Loss")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
