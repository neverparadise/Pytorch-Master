
import os
import argparse

parser = argparse.ArgumentParser(description='run_cnn')
parser.add_argument('--batch_size', type=int, default=32, help=None)
parser.add_argument('--learning_rate', type=float, default=0.001, help=None)
parser.add_argument('--epochs', type=int, default=5, help=None)
parser.add_argument('--kernel_size', type=int, default=3, help=None)
parser.add_argument('--stride', type=int, default=2, help=None)

args = parser.parse_args()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader, Dataset 
from torchvision import datasets, transforms

# Hyperparameters
batch_size = args.batch_size
learning_rate = args.learning_rate
epochs = args.epochs
kernel_size = args.kernel_size
stride = args.stride

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
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

def train(epoch, model, loss_func, train_loader, optimizer):
    model.train()
    for batch_index, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_func(y_pred, y)
        loss.backward()
        optimizer.step()
        if batch_index % 100 == 0:
            print(f'Train Epoch: {epoch+1} | Batch Status: {batch_index*len(x)}/{len(train_loader.dataset)} \
            ({100. * batch_index * batch_size / len(train_loader.dataset):.0f}% | Loss: {loss.item():.6f}')

def test(model, loss_func, test_loader):
    model.eval()
    test_loss = 0
    correct_count = 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        test_loss += loss_func(y_pred, y).item()
        pred = y_pred.data.max(1, keepdim=True)[1]
        # torch.eq : Computes element-wise equality. return counts value
        correct_count += pred.eq(y.data.view_as(pred)).cpu().sum()
    
    test_loss /= len(test_loader.dataset)
    print(f'=======================\n Test set: Average loss: {test_loss:.4f}, Accuracy: {correct_count/len(test_loader.dataset):.3}')
    
class CNN(nn.Module):
    def __init__(self, C, W, H, K, S): # 채널, 너비, 높이, 커널 사이즈, 스트라이드
        super(CNN, self).__init__()
        # nn.Module에는 이미 conv 레이어가 구현되어 있다. 
        # 배치정규화도 구현되어있고 다 구현되어있습니다. 
        self.conv1 = nn.Conv2d(C, 32, kernel_size=K, stride=S)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=K, stride=S)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=K, stride=S)
        self.bn3 = nn.BatchNorm2d(128)
        
        def conv2d_size_out(size, kernel_size=K, stride=S):
            print((size - (kernel_size - 1) - 1) // stride + 1)
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        convw = conv2d_size_out(W, K, S)
        convw = conv2d_size_out(convw, K, S)
        convw = conv2d_size_out(convw, K, S)
        
        self.linear_input_size = convw * convw * 128
        self.fc = nn.Linear(self.linear_input_size, 10)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1) # (batch_size, flatten_size)
        x = F.relu(self.fc(x))
        return F.log_softmax(x)
    
cnn = CNN(C=C, W=W, H=H, K=kernel_size, S=stride) 
cnn = cnn.to(device)
ce_loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)    

for epoch in range(epochs):
    train(epoch, cnn, ce_loss, train_loader, optimizer)

test(cnn, ce_loss, test_loader)
