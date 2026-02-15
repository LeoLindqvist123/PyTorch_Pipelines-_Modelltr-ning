import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

class CIFAR10Dataset(Dataset):
    def __init__(self, train=True):
        self.data = datasets.CIFAR10(
            root='./data',
            train=train,
            download=True,
            transform=transforms.ToTensor()
        )
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]