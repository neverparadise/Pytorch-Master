import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, C, W, H, K, S): # 채널, 너비, 높이, 커널 사이즈, 스트라이드
        super(CNN, self).__init__()
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
