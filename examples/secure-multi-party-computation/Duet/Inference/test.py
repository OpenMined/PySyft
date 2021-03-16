import torchvision
from torchvision import transforms

transform = transforms.Compose([
         transforms.ToTensor(),
         transforms.Normalize([0.1307], [0.3081]),
])

testset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
)
