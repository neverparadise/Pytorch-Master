
import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
import yaml
from models.CNN import CNN
from dataset.MNIST_LOADER import make_loader

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

parser = argparse.ArgumentParser(description='run_cnn3')
parser.add_argument('--config_path', type=str, default='./configs/cnn.yaml', help='config file path')
args = parser.parse_args()
config_path = args.config_path

with open(config_path ,'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    print(type(config))
    
batch_size = config['batch_size']
lr = config['learning_rate']
epochs = config['epochs']
kernel_size = config['kernel_size']
stride = config['stride']

train_loader, valid_loader, test_loader, shape = make_loader(batch_size)
C = shape[0]
W = shape[1]
H = shape[2]

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
cnn = CNN(C=C, W=W, H=H, K=3, S=2) 
cnn = cnn.to(device)
ce_loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr=lr)

for epoch in range(epochs):
    train(epoch, cnn, ce_loss, train_loader, optimizer)

test(cnn, ce_loss, test_loader)
