
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

def load_data(train,batch_size):
    
        """Helper function used to load the train/test data"""
        
            
        loader = torch.utils.data.DataLoader(
                       datasets.MNIST('../data', train=train, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
                       batch_size=batch_size, shuffle=True)
    
        return loader
    
def split(dataset,batch_size,split=0.2):
        
        shuffle_dataset = True
        random_seed= 42

        # Creating data indices for training and validation splits:
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(split * dataset_size))
        if shuffle_dataset :
               np.random.seed(random_seed)
               np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                           sampler=train_sampler)
        validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler)
        
        return train_loader,validation_loader
    
class NoisyDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self,dataloader,model,transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataloader=dataloader
        self.model=model
        self.transform = transform
        self.noisy_data=self.process_data()
        
            
    def process_data(self):
        
        noisy_data=[]
        
        for data,_ in self.dataloader:
            
            noisy_data.append([data,torch.tensor(self.model(data))])
            
        return noisy_data
            
    def __len__(self):
        return len(self.dataloader)

    def __getitem__(self, idx):
        
        sample=self.noisy_data[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample