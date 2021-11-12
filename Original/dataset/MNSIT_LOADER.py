from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

def make_loader(batch_size):
    train_dataset = datasets.MNIST(root='./mnist_data/', train=True, download=True, transform=transforms.ToTensor())
    valid_dataset = datasets.MNIST(root='./mnist_data/', train=False, download=True, transform=transforms.ToTensor())
    test_dataset = datasets.MNIST(root='./mnist_data/', train=False, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    vaild_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    shape = train_dataset[0][0].shape
    print(shape)
    C = shape[0]
    W = shape[1]
    H = shape[2]
    print(C, W, H)
    return train_loader, vaild_loader, test_loader, shape
