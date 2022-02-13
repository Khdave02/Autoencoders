import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
#importing mnist dataset
def trainLoader(batch_size):
    transform=transforms.ToTensor()
    train_data = datasets.MNIST(
        root="~/torch_datasets",
        train=True,                        #train data set 
        download=True,
        transform=transform              #to transform to tensors
    )
    train_loader=DataLoader(train_data, batch_size=batch_size, shuffle=True)
    return train_loader
#here batch size is no of batches in dataset
def testLoader(batch_size):
    transform=transforms.ToTensor()
    test_data = datasets.MNIST(
        root="~/torch_datasets",
        train=False,
        download=True,
        transform=transform
    )  
    test_loader=DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return test_loader

#trainLoader,testLoader = trainLoader(600),testLoader(600)
