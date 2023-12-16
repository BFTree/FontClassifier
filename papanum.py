import torch
import torch.nn as nn

class WordClassifier(nn.Module):
    def __init__(self):
        super(WordClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 32 * 32, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, 413)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x

# 创建一个虚拟输入
dummy_input = torch.randn((1, 1, 128, 128))  # 输入为灰度图像，大小为128x128

# 初始化模型并计算参数量
model = WordClassifier()
num_params = sum(p.numel() for p in model.parameters())
print(f'Number of parameters in the model: {num_params}')
