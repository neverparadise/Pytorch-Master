
import torch
from torch.utils.data import DataLoader, Dataset 
from torchvision import datasets, transforms

# 데이터로더를 리턴하는 함수를 만든다.
def make_loader(batch_size):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    train_dataset = datasets.MNIST(root='./mnist_data/', train=True, download=True, transform=transforms.ToTensor())
    valid_dataset = datasets.MNIST(root='./mnist_data/', train=False, download=True, transform=transforms.ToTensor())
    test_dataset = datasets.MNIST(root='./mnist_data/', train=False, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    shape = train_dataset[0][0].shape
    channel = shape[0]
    width = shape[1]
    height = shape[2]
    print(channel, width, height)
    return train_loader, valid_loader, test_loader, shape

#train_loader, valid_loader, test_loader, shpae = make_loader(32)
