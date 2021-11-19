import os
import yaml
import argparse

parser = argparse.ArgumentParser(description='run cnn2')
parser.add_argument('--config_path', type=str, default='./configs/', help='config_path')
parser.add_argument('--save_path', type=str, default='./weigths/', help='save_path')
parser.add_argument('--model_name', type=str, default='cnn.pth', help='model_name')
parser.add_argument('--pre_trained', type=str, default=False, help='pre_traiend')

args = parser.parse_args()
config_path = args.config_path
save_path = args.save_path
model_name = args.model_name
pre_trained = args.pre_trained
with open(config_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

batch_size = config['batch_size']
learning_rate = config['lr']
epochs = config['epochs']
kernel_size = config['kernel_size']
stride = config['stride']



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


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
        writer.add_scalar("train/loss", loss, epoch*batch_size + batch_index)
        if batch_index % 100 == 0:
            print(f'Train Epoch: {epoch+1} | Batch Status: {batch_index*len(x)}/{len(train_loader.dataset)} \
            ({100. * batch_index * batch_size / len(train_loader.dataset):.0f}% | Loss: {loss.item():.6f}')
            torch.save(model.state_dict(), save_path + model_name)

def validation(epoch, model, loss_func, valid_loader, optimizer):
    for batch_index, (x, y) in enumerate(valid_loader):
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        loss = loss_func(y_pred, y)
        writer.add_scalar("valid/loss", loss, epoch*batch_size + batch_index)

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


cnn = CNN(C=C, W=W, H=H, K=3, S=2)
cnn = cnn.to(device)
ce_loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr=0.001)

if pre_trained:
    model_dict = torch.load(save_path+model_name)
    cnn.load_state_dict(model_dict)

for epoch in range(epochs):
    train(epoch, cnn, ce_loss, train_loader, optimizer)
    valid(epoch, cnn, ce_loss, valid_loader, optimizer)
test(cnn, ce_loss, test_loader)
