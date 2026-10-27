import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ========== 1. 数据预处理与加载 ==========
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    transforms.RandomHorizontalFlip(),
])

train_path = r"D:\my_dataset\train"
test_path = r"D:\my_dataset\val"

train_dataset = datasets.ImageFolder(train_path, transform=transform)
test_dataset = datasets.ImageFolder(test_path, transform=transform)

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# ========== 2. 搭建AlexNet ==========
class MyAlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256 * 3 * 3, 2304),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(2304, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ========== 3. 设备、模型、损失、优化器 ==========
device = torch.device("cuda:0")
model1 = MyAlexNet().to(device)
model2=YOLO("C:\\Users\\Mikey\\runs\\classify\\train-4\\weights\\best.pt")
model2=model2.model
model2=model2.to(device)
model2.eval()
for p in model2.parameters():
    p.requires_grad = False
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model1.parameters(), lr=0.007, momentum=0.8)

epochs = 50
train_loss_list = []
train_acc_list = []
test_acc_list = []

# ========== 4. 训练、测试函数 ==========
def train_one_epoch(model1,model2, loader, criterion, optimizer, device):
    model1.train()
    running_loss = 0.0
    correct = 0
    total = 0
    T=3
    alpha=0.4
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs1 = model1(images)
        images_BGR=images[:,[2,1,0],:,:]
        with torch.no_grad():
            outputs2 = model2(images_BGR)[0]
        # 对每张图片，所有类别做softmax
        p_teacher = torch.softmax(outputs2 / T, dim=1)
        p_student = torch.log_softmax(outputs1 / T, dim=1)
        loss = (1-alpha)*criterion(outputs1, labels)+(alpha)*(nn.functional.kl_div(p_student, p_teacher, reduction="batchmean")*T*T)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs1.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def test_one_epoch(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    epoch_acc = correct / total
    return epoch_acc

# ========== 5. 主训练循环 ==========
for epoch in range(epochs):
    train_loss, train_acc = train_one_epoch(model1, model2,train_loader, criterion, optimizer, device)
    test_acc = test_one_epoch(model1, test_loader, device)

    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    print(f"Epoch [{epoch+1}/{epochs}] | Loss:{train_loss:.4f} | Train Acc:{train_acc:.4f} | Test Acc:{test_acc:.4f}")

# ========== 6. 绘图 ==========
plt.figure(figsize=(8, 6))
x = list(range(1, epochs + 1))
plt.plot(x, train_loss_list, label="train loss", color="blue")
plt.plot(x, train_acc_list, label="train acc", color="magenta", linestyle="--")
plt.plot(x, test_acc_list, label="test acc", color="green", linestyle="--")
plt.xlabel("epoch")
plt.legend()
plt.grid(True)
plt.savefig("D:\\my_dataset\\result6.png",dpi=300)




