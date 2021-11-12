import wandb
from subprocess import call
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from dataset import MNSIT_LOADER
from models.CNN import CNN
import os

#call(["wandb", "login", "발급받은 API 키 입력"])

call(["wandb", "login", ''])

wandb.init(project="flipped_school", entity="neverparadise")

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
wandb.config = {
  "learning_rate": 0.001,
  "epochs": 5,
  "batch_size": 64
}

batch_size = wandb.config['batch_size']
lr = wandb.config['learning_rate']
epochs = wandb.config['epochs']
save_path = os.curdir + '/weights/'
model_name = 'cnn.pth'
train_loader, valid_loader, test_loader, shape = MNSIT_LOADER.make_loader(batch_size)

def train(epoch, model, loss_func, train_loader, valid_loader, optimizer):
    model.train()
    for batch_index, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_func(y_pred, y)
        loss.backward()
        optimizer.step()
        wandb.log({"train_loss": loss})


        if batch_index % 100 == 0:
            print(f'Train Epoch: {epoch+1} | Batch Status: {batch_index*len(x)}/{len(train_loader.dataset)} \
            ({100. * batch_index * batch_size / len(train_loader.dataset):.0f}% | Loss: {loss.item():.6f}')
            torch.save(model.state_dict(), save_path + model_name)

def validation(epoch, model, loss_func, valid_loader):
    for batch_index, (x, y) in enumerate(valid_loader):
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        val_loss = loss_func(y_pred, y)
        wandb.log({"val_loss": val_loss})

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

cnn = CNN(C=shape[0], W=shape[1], H=shape[2], K=3, S=2)
cnn = cnn.to(device)
ce_loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr=lr)

for epoch in range(epochs):
    train(epoch, cnn, ce_loss, train_loader, valid_loader, optimizer)
    validation(epoch, cnn, ce_loss, valid_loader)

test(cnn, ce_loss, test_loader)
