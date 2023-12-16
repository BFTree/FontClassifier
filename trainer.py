import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
import os


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
        self.dropout = nn.Dropout(0.0)
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

data_dir = 'WorkData'
img_size = 128
rotation_degree = 90

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize([img_size,img_size]),
        transforms.RandomRotation(degrees=(-rotation_degree, rotation_degree)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.485], [0.229])
    ]),
    'val': transforms.Compose([
        transforms.Resize([img_size,img_size]),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.485], [0.229])
    ]),
    'test': transforms.Compose([
        transforms.Resize([img_size,img_size]),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.485], [0.229])
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val', 'test']}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = WordClassifier()
net.to(device)

criterion = nn.NLLLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
batch_size = 64
dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in ['train', 'val', 'test']}

num_epochs = 50
best_val_accuracy = 0.0
best_model_path = 'models/best_model.pth'  

stage_count = 0
stage_num = 0
correct_top5 = 0

with open('result.txt', 'w') as result_file:
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()

            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in tqdm(dataloaders[phase], desc=f'Epoch {epoch + 1}/{num_epochs} ({phase})'):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                # 计算模型输出的概率分布
                probabilities = F.softmax(predicted, dim=1)

                # 获取前五个预测的类别
                top5_predictions = torch.topk(probabilities, k=5, dim=1)[1]

                # 检查真实标签是否在前五个预测中
                correct_top5 += labels.view(-1, 1).expand_as(top5_predictions).eq(top5_predictions).sum().item()
                stage_count += 1
                if stage_count > 2500:
                    temp_accuracy = correct_top5 / total
                    print(f'stage:{stage_num} accuracy: {temp_accuracy:.4f}')
                    stage_count = 0
                    stage_num += 1

            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_accuracy = correct_top5 / total

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Accuracy: {epoch_accuracy:.4f}')

            torch.save(net.state_dict(), f'models/model_epoch_{epoch + 1}.pth')

            result_file.write(f'{phase.capitalize()} Epoch {epoch + 1}: Loss={epoch_loss:.4f}, Accuracy={epoch_accuracy:.4f}\n')

            if phase == 'val' and epoch_accuracy > best_val_accuracy:
                best_val_accuracy = epoch_accuracy
                torch.save(net.state_dict(), best_model_path)


net.load_state_dict(torch.load(best_model_path))
net.eval()

test_correct = 0
test_total = 0

for inputs, labels in tqdm(dataloaders['test'], desc='Testing'):
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = net(inputs)
    _, predicted = torch.max(outputs, 1)
    test_total += labels.size(0)
    test_correct += (predicted == labels).sum().item()

test_accuracy = test_correct / test_total
print(f'Test Accuracy: {test_accuracy:.4f}')

with open('result.txt', 'a') as f:
    f.write(f'Test Accuracy: {test_accuracy:.4f}\n')
