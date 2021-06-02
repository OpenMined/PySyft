# future
from __future__ import print_function

# stdlib
import argparse
import os
import random

# third party
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    required=True,
    help="cifar10 | lsun | mnist |imagenet | folder | lfw | fake",
)
parser.add_argument("--data_folder", required=False, help="path to dataset")
parser.add_argument(
    "--data_workers", type=int, help="number of data loading data_workers", default=2
)
parser.add_argument("--batch_size", type=int, default=64, help="input batch size")
parser.add_argument(
    "--image_size",
    type=int,
    default=64,
    help="the height / width of the input image to network",
)
parser.add_argument("--nz", type=int, default=100, help="size of the latent z vector")
parser.add_argument("--ngf", type=int, default=64)
parser.add_argument("--ndf", type=int, default=64)
parser.add_argument(
    "--n_iter", type=int, default=25, help="number of epochs to train for"
)
parser.add_argument(
    "--lr", type=float, default=0.0002, help="learning rate, default=0.0002"
)
parser.add_argument(
    "--beta1", type=float, default=0.5, help="beta1 for adam. default=0.5"
)
parser.add_argument("--cuda", action="store_true", help="enables cuda")
parser.add_argument(
    "--dry-run", action="store_true", help="check a single training cycle works"
)
parser.add_argument("--n_gpu", type=int, default=1, help="number of GPUs to use")
parser.add_argument(
    "--net_generator", default="", help="path to net_generator (to continue training)"
)
parser.add_argument(
    "--net_discriminator",
    default="",
    help="path to net_discriminator (to continue training)",
)
parser.add_argument(
    "--out_folder", default=".", help="folder to output images and model checkpoints"
)
parser.add_argument("--manual_seed", type=int, help="manual seed")
parser.add_argument(
    "--classes",
    default="bedroom",
    help="comma separated list of classes for the lsun data set",
)

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.out_folder)
except OSError:
    pass

if opt.manual_seed is None:
    opt.manual_seed = random.randint(1, 10000)
print("Random Seed: ", opt.manual_seed)
random.seed(opt.manual_seed)
torch.manual_seed(opt.manual_seed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.data_folder is None and str(opt.dataset).lower() != "fake":
    raise ValueError(f'`data_folder` parameter is required for dataset "{opt.dataset}"')

if opt.dataset in ["imagenet", "folder", "lfw"]:
    # folder dataset
    dataset = dset.ImageFolder(
        root=opt.data_folder,
        transform=transforms.Compose(
            [
                transforms.Resize(opt.image_size),
                transforms.CenterCrop(opt.image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )
    nc = 3
elif opt.dataset == "lsun":
    classes = [c + "_train" for c in opt.classes.split(",")]
    dataset = dset.LSUN(
        root=opt.data_folder,
        classes=classes,
        transform=transforms.Compose(
            [
                transforms.Resize(opt.image_size),
                transforms.CenterCrop(opt.image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )
    nc = 3
elif opt.dataset == "cifar10":
    dataset = dset.CIFAR10(
        root=opt.data_folder,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize(opt.image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )
    nc = 3

elif opt.dataset == "mnist":
    dataset = dset.MNIST(
        root=opt.data_folder,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize(opt.image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        ),
    )
    nc = 1

elif opt.dataset == "fake":
    dataset = dset.FakeData(
        image_size=(3, opt.image_size, opt.image_size), transform=transforms.ToTensor()
    )
    nc = 3

assert dataset
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batch_size,
    shuffle=True,
    num_data_workers=int(opt.data_workers),
)

device = torch.device("cuda:0" if opt.cuda else "cpu")
n_gpu = int(opt.n_gpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)


# custom weights initialization called on net_generator and net_discriminator
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


class Generator(nn.Module):
    def __init__(self, n_gpu):
        super(Generator, self).__init__()
        self.n_gpu = n_gpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.n_gpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.n_gpu))
        else:
            output = self.main(input)
        return output


net_generator = Generator(n_gpu).to(device)
net_generator.apply(weights_init)
if opt.net_generator != "":
    net_generator.load_state_dict(torch.load(opt.net_generator))
print(net_generator)


class Discriminator(nn.Module):
    def __init__(self, n_gpu):
        super(Discriminator, self).__init__()
        self.n_gpu = n_gpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):
        if input.is_cuda and self.n_gpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.n_gpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


net_discriminator = Discriminator(n_gpu).to(device)
net_discriminator.apply(weights_init)
if opt.net_discriminator != "":
    net_discriminator.load_state_dict(torch.load(opt.net_discriminator))
print(net_discriminator)

criterion = nn.BCELoss()

fixed_noise = torch.randn(opt.batch_size, nz, 1, 1, device=device)
real_label = 1
fake_label = 0

# setup optimizer
optim_discriminator = optim.Adam(
    net_discriminator.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)
)
optim_generator = optim.Adam(
    net_generator.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)
)

if opt.dry_run:
    opt.n_iter = 1

for epoch in range(opt.n_iter):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        net_discriminator.zero_grad()
        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)
        label = torch.full(
            (batch_size,), real_label, dtype=real_cpu.dtype, device=device
        )
        output = net_discriminator(real_cpu)
        err_d_real = criterion(output, label)
        err_d_real.backward()
        d_x = output.mean().item()

        # train with fake
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = net_generator(noise)
        label.fill_(fake_label)
        output = net_discriminator(fake.detach())
        err_d_fake = criterion(output, label)
        err_d_fake.backward()
        d_g_z1 = output.mean().item()
        err_discrimantor = err_d_real + err_d_fake
        optim_discriminator.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        net_generator.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        output = net_discriminator(fake)
        err_generator = criterion(output, label)
        err_generator.backward()
        d_g_z2 = output.mean().item()
        optim_generator.step()

        print(
            "[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f"
            % (
                epoch,
                opt.n_iter,
                i,
                len(dataloader),
                err_discrimantor.item(),
                err_generator.item(),
                d_x,
                d_g_z1,
                d_g_z2,
            )
        )
        if i % 100 == 0:
            vutils.save_image(
                real_cpu, f"{opt.out_folder}/real_samples.png", normalize=True
            )
            fake = net_generator(fixed_noise)
            vutils.save_image(
                fake.detach(),
                "%s/fake_samples_epoch_%03d.png" % (opt.out_folder, epoch),
                normalize=True,
            )

        if opt.dry_run:
            break
    # do checkpointing
    torch.save(
        net_generator.state_dict(),
        "%s/net_generator_epoch_%d.pth" % (opt.out_folder, epoch),
    )
    torch.save(
        net_discriminator.state_dict(),
        "%s/net_discriminator_epoch_%d.pth" % (opt.out_folder, epoch),
    )
